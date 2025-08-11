import logging
from tqdm import tqdm
import copy
import os
import errno
import os.path as osp
from torch import optim
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from models.base import BaseLearner
from utils.inc_net import FDDR_CIFAR
from utils.toolkit import tensor2numpy
from utils.estimator_cv import *

softmax = nn.Softmax(dim=1)
T = 2


class FDDR_CIFAR_Learner(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FDDR_CIFAR(args['convnet_type'], False)
        self._evolved_protos = []
        self.CoVariance = None
        self.Ave = None
        self.old_covariances = None
        self.new_covariances = None
        self.old_averages = None
        self.new_averages = None

    def after_task(self):
        self._known_classes = self._total_classes
        if self.args["test"] is False:
            self._old_network = self._network.copy().freeze()
            save_checkpoint({
                'state_dict': self._network.state_dict(),
                'seed': self.args['seed'],
                'task': self._cur_task,
            }, fpath=osp.join(self.args['save_path'],'checkpoint_' + self.args["dataset"] + '_' + str(self.args['seed']) + '_' + 'task' + str(self._cur_task + 1) + '.pth.tar'))
            logging.info('Exemplar size: {}'.format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        # Loader
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train')
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True, num_workers=self.args["num_workers"])

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args["num_workers"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def incremental_test(self, data_manager):
        self._cur_task += 1
        self.data_manager = data_manager
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes, self._cur_task)

        logging.info('Evaluation on {}-{}'.format(self._known_classes, self._total_classes))

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False, num_workers=self.args["num_workers"])
        checkpoint = torch.load( "./pretrain/" + 'checkpoint_' + self.args["dataset"] + '_' + str(self.args['seed']) + '_' + 'task' + str(self._cur_task + 1) + '.pth.tar')
        self._network.load_state_dict(checkpoint["state_dict"])

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._network.to(self._device)

        self.build_rehearsal_memory(data_manager, self.samples_per_class)


    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
            )), momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["init_weight_decay"])

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["init_epochs"])
            self._init_train(train_loader, test_loader, optimizer, scheduler)
            self._compute_old_statistics(train_loader)
        else:
            # Model Adaption
            self.model_a = FDDR_CIFAR(self.args['convnet_type'], False)
            self.model_a.update_fc(self._total_classes, self._cur_task)
            if len(self._multiple_gpus) > 1:
                self.model_a = nn.DataParallel(self.model_a, self._multiple_gpus)
            self.model_a.to(self._device)
            if len(self._multiple_gpus) > 1:
                self.model_a.module.convnet.load_state_dict(copy.deepcopy(self._network.module.convnet.state_dict()))
                self.model_a.module.copy_fc(copy.deepcopy(self._network.module.fc))
            else:
                self.model_a.convnet.load_state_dict(copy.deepcopy(self._network.convnet.state_dict()))
                self.model_a.copy_fc(copy.deepcopy(self._network.fc))

            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model_a.parameters(
            )), lr=self.args["lr"], momentum=0.9, weight_decay=self.args["weight_decay"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=self.args["adaptation_epochs"])
            self._model_adaptation(train_loader, test_loader, optimizer, scheduler)
            self._model_fddr(test_loader)



    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                features, logits, featuremaps = self._network(inputs, targets, 0, 0, 0)

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self.args["init_epochs"], losses / len(train_loader), train_acc,
                    test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self.args["init_epochs"], losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

            logging.info(info)

    def _model_adaptation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["adaptation_epochs"]))
        for _, epoch in enumerate(prog_bar):
            self.model_a.train()
            losses = 0.
            correct, total = 0, 0
            for _, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                _, logits, featuremaps = self.model_a(inputs, targets, 0, 0, 0)

                loss_clf = F.cross_entropy(logits[:, self._known_classes:], (targets - self._known_classes))
                loss = loss_clf

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self.args["adaptation_epochs"], losses / len(train_loader), train_acc,
                    test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, self.args["adaptation_epochs"], losses / len(train_loader), train_acc)
            prog_bar.set_description(info)

        logging.info(info)

    def _train_stage1(self, train_loader, model, optimizer, scheduler):
        model.train()
        criterion_mcv = CriterionMCV(self._network.feature_dim, self._total_classes, self._device)
        losses = 0.
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            features, y_f, fmaps = model(inputs, targets, 0, 0, 0)
            with torch.no_grad():
                _, new_logits, new_fmaps = self.model_a(inputs, targets, 0, 0, 0)
                _, old_logits, old_fmaps = self._old_network(inputs, targets, 0, 0, 0)

            loss_cls = criterion_mcv(features, y_f, targets)

            cv = criterion_mcv.get_cv()
            contrastive_covariances = cv.clone()[0: self._known_classes]

            ave = criterion_mcv.get_average()
            contrastive_averages = ave.clone()[0: self._known_classes]

            cov_diff = self.old_covariances - contrastive_covariances
            cov_loss_per_class = torch.norm(cov_diff, p='fro', dim=(1, 2))

            mean_diff = self.old_averages - contrastive_averages
            mean_loss_per_class = torch.norm(mean_diff, p=2, dim=1)

            average_cov_loss = cov_loss_per_class.mean()
            average_mean_loss = mean_loss_per_class.mean()

            loss_reg = average_cov_loss + average_mean_loss
            loss_kd_old = _KD_loss(
                y_f[:, : self._known_classes],
                old_logits[:, : self._known_classes],
                T,
            )
            loss_kd_new = _KD_loss(
                y_f[:, self._known_classes:],
                new_logits[:, self._known_classes:],
                T,
            )

            if self.args['init_cls'] == 50:
                spatial_loss = pod_spatial_loss(fmaps, old_fmaps) * self.factor * 5
                total_loss = loss_cls + self.args["beta"] * loss_reg + self.args["alpha"] * loss_kd_old +  loss_kd_new + self.args["spatial"] * spatial_loss
            else:
                total_loss = loss_cls + self.args["beta"] * loss_reg + self.args["alpha"] * loss_kd_old +  loss_kd_new

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses += total_loss.item()

            _, preds = torch.max(y_f, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        scheduler.step()
        return losses / len(train_loader), train_acc

    def _train_stage2(self, train_loader, model, optimizer, scheduler):
        model.train()
        losses = 0.
        correct, total = 0, 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            predictions = model.fc(inputs)['logits']
            loss_cls = F.cross_entropy(predictions, targets)
            optimizer.zero_grad()
            loss_cls.backward()
            optimizer.step()
            losses += loss_cls.item()

            _, preds = torch.max(predictions, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)

        train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        scheduler.step()
        return losses / len(train_loader), train_acc

    def _train_stage3(self, train_loader, model, optimizer, scheduler, logits_adjustment):
        model.train()
        criterion_mcv = CriterionMCV(self._network.feature_dim, self._total_classes, self._device)
        losses = 0.
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            features, y_f, fmaps = model(inputs, targets, 0, 0, 0)
            with torch.no_grad():
                _, new_logits, new_fmaps = self.model_a(inputs, targets, 0, 0, 0)
                _, old_logits, old_fmaps = self._old_network(inputs, targets, 0, 0, 0)
            y_f = y_f + logits_adjustment
            loss_cls = criterion_mcv(features, y_f, targets)
            cv = criterion_mcv.get_cv()
            contrastive_covariances = cv.clone()[0:self._known_classes]
            ave = criterion_mcv.get_average()
            contrastive_averages = ave.clone()[0:self._known_classes]

            cov_diff = self.CoVariance[0:self._known_classes] - contrastive_covariances
            cov_loss_per_class = torch.norm(cov_diff, p='fro', dim=(1, 2))

            mean_diff = self.Ave[0:self._known_classes] - contrastive_averages
            mean_loss_per_class = torch.norm(mean_diff, p=2, dim=1)

            average_cov_loss = cov_loss_per_class.mean()
            average_mean_loss = mean_loss_per_class.mean()

            loss_reg = average_cov_loss + average_mean_loss
            loss_kd_old = _KD_loss(
                y_f[:, : self._known_classes],
                old_logits[:, : self._known_classes],
                T,
            )
            loss_kd_new = _KD_loss(
                y_f[:, self._known_classes:],
                new_logits[:, self._known_classes:],
                T,
            )
            if self.args['init_cls'] == 50:
                spatial_loss = pod_spatial_loss(fmaps, old_fmaps) * self.factor * 5
                total_loss = loss_cls + self.args["beta"] * loss_reg + self.args["alpha"] * loss_kd_old +  loss_kd_new + \
                             self.args["spatial"] * spatial_loss
            else:
                total_loss = loss_cls + self.args["beta"] * loss_reg + self.args["alpha"] * loss_kd_old +  loss_kd_new

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            losses += total_loss.item()

            _, preds = torch.max(y_f, dim=1)
            correct += preds.eq(targets.expand_as(preds)).cpu().sum()
            total += len(targets)


        train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
        scheduler.step()
        return losses / len(train_loader), train_acc


    def _calculate_similarity(self, prototype):
        norms = torch.norm(prototype, dim=1)
        dot_product_matrix = torch.mm(prototype, prototype.T)
        norm_matrix = torch.ger(norms, norms)
        similarity_matrix = dot_product_matrix / norm_matrix
        similarity_matrix = (similarity_matrix + 1) / 2
        row_sums = similarity_matrix.sum(dim=1, keepdim=True)
        similarity_matrix_norm = similarity_matrix / row_sums
        return similarity_matrix_norm


    def _compute_statistics(self, loader):
        vectors, targets = self._extract_vectors(loader)
        feature_num = self._network.feature_dim
        class_num = self._total_classes
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).to(self._device)
        self.Ave = torch.zeros(class_num, feature_num).to(self._device)

        for c in range(self._total_classes):
            idx = np.where(targets == c)[0]
            if len(idx) == 0:
                continue

            class_vectors = vectors[idx]
            mean_vector = np.mean(class_vectors, axis=0)
            cov_matrix = np.cov(class_vectors, rowvar=False)
            self.Ave[c] = torch.from_numpy(mean_vector).float().to(self._device)
            self.CoVariance[c] = torch.from_numpy(cov_matrix).float().to(self._device)
        return targets

    def _distribution_reconstruction(self, loader, gamma, out_new, kg, targets):
        augment_vectors, augment_targets = self._extract_vectors_aug(loader, gamma, out_new)
        cv_matrix_temp = self.CoVariance.view(self.CoVariance.size(0), -1).to(self._device)
        kg = kg.to(self._device)
        cv_var_new = torch.matmul(kg[:self._known_classes], cv_matrix_temp).view(self._known_classes, self._network.feature_dim, -1)
        cv_var = self.CoVariance.to(self._device)
        cv_var_new = cv_var_new.to(self._device)
        new_cv = torch.cat((cv_var_new, cv_var[self._known_classes:]), 0)
        self.CoVariance = self.args["lamda"] * new_cv
        self.Ave[:self._known_classes] = self.Ave[:self._known_classes] + gamma * out_new[:self._known_classes]
        targets = np.array(targets)
        target_counts = []
        for c in range(self._known_classes, self._total_classes):
            count = np.sum(targets == c)
            target_counts.append(count)
        target_count = min(target_counts)

        additional_vectors = []
        additional_targets = []
        for c in range(self._known_classes):
            current_count = np.sum(targets == c)
            if current_count < target_count:
                sample_num = target_count - current_count

                mean_np = self.Ave[c].detach().cpu().numpy()
                cov_np = self.CoVariance[c].detach().cpu().numpy()
                sampled_features = np.random.multivariate_normal(mean=mean_np,
                                                                 cov=cov_np,
                                                                 size=sample_num)
                additional_vectors.append(sampled_features)
                additional_targets.append(np.full(sample_num, c))

        additional_vectors = np.concatenate(additional_vectors, axis=0)
        additional_targets = np.concatenate(additional_targets, axis=0)
        balanced_vectors = np.concatenate([augment_vectors, additional_vectors], axis=0)
        balanced_targets = np.concatenate([augment_targets, additional_targets], axis=0)

        balanced_vectors = torch.from_numpy(balanced_vectors).float()
        balanced_targets = torch.from_numpy(balanced_targets).long()

        dataset = torch.utils.data.TensorDataset(balanced_vectors, balanced_targets)
        balanced_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=self.args['batch_size'],
                                                      shuffle=True,
                                                      num_workers=self.args["num_workers"])
        return balanced_loader

    def _compute_old_statistics(self, loader):
        vectors, targets = self._extract_vectors(loader)
        feature_num = self._network.feature_dim
        class_num = self._total_classes
        self.old_covariances = torch.zeros(class_num, feature_num, feature_num).to(self._device)
        self.old_averages = torch.zeros(class_num, feature_num).to(self._device)

        for c in range(self._total_classes):
            idx = np.where(targets == c)[0]
            if len(idx) == 0:
                continue
            class_vectors = vectors[idx]
            mean_vector = np.mean(class_vectors, axis=0)

            cov_matrix = np.cov(class_vectors, rowvar=False)
            self.old_averages[c] = torch.from_numpy(mean_vector).float().to(self._device)
            self.old_covariances[c] = torch.from_numpy(cov_matrix).float().to(self._device)
            self._evolved_protos.append(torch.from_numpy(mean_vector).float().to(self._device))

    def _update_statistics(self, loader):
        vectors, targets = self._extract_vectors(loader)
        feature_num = self._network.feature_dim
        class_num = self._total_classes
        self.new_covariances = torch.zeros(class_num, feature_num, feature_num).to(self._device)
        self.new_averages = torch.zeros(class_num, feature_num).to(self._device)
        for c in range(self._known_classes, self._total_classes):
            idx = np.where(targets == c)[0]
            if len(idx) == 0:
                continue
            class_vectors = vectors[idx]

            mean_vector = np.mean(class_vectors, axis=0)
            cov_matrix = np.cov(class_vectors, rowvar=False)
            self.new_averages[c] = torch.from_numpy(mean_vector).float().to(self._device)
            self.new_covariances[c] = torch.from_numpy(cov_matrix).float().to(self._device)
            self._evolved_protos.append(torch.from_numpy(mean_vector).float().to(self._device))

        # compute the feature drift between old model and new model & update evolved prototype
        for class_idx in range(0, self._known_classes):
            class_mask = self._targets_memory == class_idx
            x_class = self._data_memory[class_mask]
            y_class = self._targets_memory[class_mask]
            idx_dataset = self.data_manager.get_dataset([], source='train', mode='test',
                                                        appendent=(x_class, y_class))

            idx_loader = DataLoader(idx_dataset, batch_size=self.args['batch_size'], shuffle=False)

            if idx_loader is not None:
                vectors_old = self._extract_vectors_adv(idx_loader, old=True)[0]
                vectors = self._extract_vectors_adv(idx_loader)[0]
            MU = np.asarray(self._evolved_protos[class_idx].unsqueeze(0).cpu())
            gap = np.mean(vectors - vectors_old, axis=0)
            MU += gap
            self._evolved_protos[class_idx] = torch.tensor(MU).squeeze(0).to(self._device)

    def _model_fddr(self, test_loader):
        if self.args['init_cls'] == 50:
            if self._cur_task == 0:
                self.factor = 0
            else:
                self.factor = math.sqrt(self._total_classes / (self._total_classes - self._known_classes))

        train_dataset = self.data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                      source='train',
                                                      mode='train', appendent=self._get_memory())
        imbalanced_train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True,
                                             num_workers=self.args["num_workers"])
        optimizer = optim.SGD(filter(
            lambda p: p.requires_grad, self._network.parameters()), lr=self.args["lr"], momentum=0.9)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["epoch_stage1"])
        self.model_a.eval()
        self._old_network.eval()

        epoch_total = self.args["epoch_stage1"] + self.args["epoch_stage2"] + self.args["epoch_stage3"]
        prog_bar = tqdm(range(epoch_total))
        logits_adjustment = Logit_adjustment(self, imbalanced_train_loader, self.args['tro'])
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            if epoch < self.args['epoch_stage1']:
                losses, train_acc = self._train_stage1(imbalanced_train_loader, self._network, optimizer, scheduler)
            elif epoch < (self.args['epoch_stage1'] + self.args["epoch_stage2"]):
                if epoch == self.args['epoch_stage1']:
                    evolved_prototype = []
                    for class_idx in range(0, self._known_classes):
                        class_mask = self._targets_memory == class_idx
                        x_class = self._data_memory[class_mask]
                        y_class = self._targets_memory[class_mask]

                        idx_dataset = self.data_manager.get_dataset([], source='train', mode='test',
                                                                    appendent=(x_class, y_class))

                        idx_loader = DataLoader(idx_dataset, batch_size=self.args['batch_size'], shuffle=False)

                        if idx_loader is not None:
                            vectors_old = self._extract_vectors_adv(idx_loader, old=True)[0]
                            vectors = self._extract_vectors_adv(idx_loader)[0]

                        MU = np.asarray(self._evolved_protos[class_idx].unsqueeze(0).cpu())
                        gap = np.mean(vectors - vectors_old, axis=0)
                        MU += gap
                        evolved_prototype.append(torch.tensor(MU).squeeze(0).to(self._device))

                    targets = self._compute_statistics(imbalanced_train_loader)

                    evolved_prototype = torch.stack(evolved_prototype).float().to(self._device)
                    joint_prototype = copy.deepcopy(self.Ave)
                    joint_prototype[:self._known_classes] = self.args["eta"] * evolved_prototype[:self._known_classes] + (1 - self.args["eta"]) * self.Ave[:self._known_classes]

                    joint_prototype_matrix = copy.deepcopy(joint_prototype)
                    joint_prototype_matrix = joint_prototype_matrix.to(torch.float32).to(self._device)
                    kg = self._calculate_similarity(joint_prototype_matrix)
                    kg = torch.tensor(kg).to(self._device)
                    kg_diag = torch.diag(kg)

                    kg = kg.to(torch.float32).to(self._device)
                    zero = torch.zeros_like(kg).to(torch.float32).to(self._device)
                    zero[:self._known_classes, -self.args['increment']:] = kg[:self._known_classes, -self.args['increment']:]


                    kg = zero
                    for i in range(len(kg_diag)):
                        kg[i, i] = kg_diag[i]
                    row_sum = kg.sum(dim=1, keepdim=True)
                    row_sum = row_sum + 1e-8

                    kg = kg / row_sum
                    # use kg to get reasoning prototype
                    out_new = torch.matmul(kg, joint_prototype)
                    # subtract itself
                    out_new = out_new - torch.matmul(torch.diag(kg_diag), joint_prototype)


                    logging.info('Freezing feature weights except for self attention weights (if exist).')
                    for param_name, param in self._network.named_parameters():
                        logging.info("param_name:{}".format(param_name))
                        if 'fc' not in param_name:
                            param.requires_grad = False
                        logging.info('{}|{}'.format(param_name, param.requires_grad))

                    optimizer2 = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                    )), lr=self.args["lr"], momentum=0.9, weight_decay=self.args["weight_decay"])

                    scheduler2 = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer2, T_max=(self.args["epoch_stage2"]))

                    balanced_feature_loader = self._distribution_reconstruction(imbalanced_train_loader, self.args["gamma"], out_new, kg,
                                                                    targets)
                losses, train_acc = self._train_stage2(balanced_feature_loader, self._network, optimizer2, scheduler2)
            else:
                # finetuning
                for param_name, param in self._network.named_parameters():
                    param.requires_grad = True

                if epoch == (self.args['epoch_stage1'] + self.args["epoch_stage2"]):
                    optimizer3 = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters(
                    )), lr=0.1, momentum=0.9)
                    scheduler3 = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer=optimizer3, T_max=(self.args["epoch_stage3"]))

                losses, train_acc = self._train_stage3(imbalanced_train_loader, self._network, optimizer3, scheduler3, logits_adjustment)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, epoch_total, losses, train_acc,
                    test_acc)
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch + 1, epoch_total, losses, train_acc)
            prog_bar.set_description(info)


        self._update_statistics(imbalanced_train_loader)
        self.old_covariances = torch.cat((self.old_covariances, self.new_covariances[self._known_classes:]), dim=0)
        self.old_averages = torch.cat((self.old_averages, self.new_averages[self._known_classes:]), dim=0)
        for param_name, param in self._network.named_parameters():
            param.requires_grad = True


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


def crossEntropy(softmax, logit, label, weight, num_classes, known_class):
    target = F.one_hot(label, num_classes)
    loss = - (weight * (target * torch.log(softmax(logit) + 1e-7)).sum(dim=1)).sum()
    return loss


def Logit_adjustment(self, train_loader, tro):
    """compute the base probabilities"""

    label_freq = {}
    for i, (_, inputs, target) in enumerate(train_loader):
        target = target.to(self._device)
        for j in target:
            key = int(j.item())
            label_freq[key] = label_freq.get(key, 0) + 1
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** tro + 1e-12)
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(self._device)

    return adjustments


def save_checkpoint(state, fpath=''):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def pod_spatial_loss(old_fmaps, fmaps, normalize=True):
    '''
    a, b: list of [bs, c, w, h]
    '''
    loss = torch.tensor(0.).to(fmaps[0].device)
    for i, (a, b) in enumerate(zip(old_fmaps, fmaps)):
        assert a.shape == b.shape, 'Shape error'

        a = torch.pow(a, 2)
        b = torch.pow(b, 2)

        a_h = a.sum(dim=3).view(a.shape[0], -1)  # [bs, c*w]
        b_h = b.sum(dim=3).view(b.shape[0], -1)  # [bs, c*w]
        a_w = a.sum(dim=2).view(a.shape[0], -1)  # [bs, c*h]
        b_w = b.sum(dim=2).view(b.shape[0], -1)  # [bs, c*h]

        a = torch.cat([a_h, a_w], dim=-1)
        b = torch.cat([b_h, b_w], dim=-1)

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        layer_loss = torch.mean(torch.frobenius_norm(a - b, dim=-1))
        loss += layer_loss

    return loss / len(fmaps)