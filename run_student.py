#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import argparse
import time
import torch

torch.set_num_threads(1)
import pickle
from student_model.models import MLP_Student
from utils.train import *
from utils.load_data import *
from utils.log import TrainLogger
from models.losses import *
from models import trainer
from models.model import D2STGNN
import yaml
import setproctitle


def main(**kwargs):
    set_config(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='METR-LA', help='Dataset name.')
    # parser.add_argument('--dataset', type=str, default='PEMS-BAY', help='Dataset name.')
    # parser.add_argument('--dataset', type=str, default='PEMS04', help='Dataset name.')
    # parser.add_argument('--dataset', type=str, default='PEMS08', help='Dataset name.')
    args = parser.parse_args()

    config_path = "configs/" + args.dataset + ".yaml"

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['data_args']['data_dir']
    dataset_name = config['data_args']['data_dir'].split("/")[-1]

    device = torch.device(config['start_up']['device'])
    save_path = 'output/' + config['start_up']['model_name'] + "_" + dataset_name + ".pt"  # the best model
    save_path_resume = 'output/' + config['start_up'][
        'model_name'] + "_" + dataset_name + "_resume.pt"  # the resume model
    load_pkl = config['start_up']['load_pkl']
    model_name = config['start_up']['model_name']

    model_name = config['start_up']['model_name']
    setproctitle.setproctitle("{0}.{1}@S22".format(model_name, dataset_name))

    # ========================== load dataset, adjacent matrix, node embeddings ====================== #
    if load_pkl:
        t1 = time.time()
        dataloader = pickle.load(open('output/dataloader_' + dataset_name + '.pkl', 'rb'))
        t2 = time.time()
        print("Load dataset: {:.2f}s...".format(t2 - t1))
    else:
        t1 = time.time()
        batch_size = config['model_args']['batch_size']
        dataloader = load_dataset(data_dir, batch_size, batch_size, batch_size, dataset_name)
        # pickle.dump(dataloader, open('output/dataloader_' + dataset_name + '.pkl', 'wb'))
        t2 = time.time()
        print("Load dataset: {:.2f}s...".format(t2 - t1))
    scaler = dataloader['scaler']

    if dataset_name == 'PEMS04' or dataset_name == 'PEMS08':  # traffic flow
        _min = pickle.load(open("datasets/{0}/min.pkl".format(dataset_name), 'rb'))
        _max = pickle.load(open("datasets/{0}/max.pkl".format(dataset_name), 'rb'))
    else:
        _min = None
        _max = None

    t1 = time.time()
    adj_mx, adj_ori = load_adj(config['data_args']['adj_data_path'], config['data_args']['adj_type'])
    t2 = time.time()
    print("Load adjacent matrix: {:.2f}s...".format(t2 - t1))

    # ================================ Hyper Parameters ================================= #
    # model parameters
    model_args = config['model_args']
    # print(model_args)
    model_args['device'] = device
    model_args['num_nodes'] = adj_mx[0].shape[0]
    model_args['adjs'] = [torch.tensor(i).to(device) for i in adj_mx]
    model_args['adjs_ori'] = torch.tensor(adj_ori).to(device)
    model_args['dataset'] = dataset_name

    # training strategy parametes
    optim_args = config['optim_args']
    optim_args['cl_steps'] = optim_args['cl_epochs'] * len(dataloader['train_loader'])
    optim_args['warm_steps'] = optim_args['warm_epochs'] * len(dataloader['train_loader'])
    # ============================= Model and Trainer ============================= #
    # log
    logger = TrainLogger(model_name, dataset_name)
    logger.print_model_args(model_args, ban=['adjs', 'adjs_ori', 'node_emb'])
    logger.print_optim_args(optim_args)

    # init the model
    model = D2STGNN(**model_args).to(device)
    # print(model)
    # get a trainer
    engine = trainer(scaler, model,model, **optim_args)
    early_stopping = EarlyStopping(optim_args['patience'], save_path)

    # begin training:
    train_time = []  # training time
    val_time = []  # validate time

    print("Whole trainining iteration is " + str(len(dataloader['train_loader'])))

    # training init: resume model & load parameters
    mode = config['start_up']['mode']
    assert mode in ['test', 'resume', 'scratch']
    resume_epoch = 0
    if 1==1:
        teacher_model = load_model(model, save_path)  # resume best

    batch_num = resume_epoch * len(dataloader['train_loader'])  # batch number (maybe used in schedule sampling)

    engine.set_resume_lr_and_cl(resume_epoch, batch_num)
    student_model=1
    # teacher_model
    engine.test(teacher_model,save_path_resume, device, dataloader, scaler, model_name, save=False, _max=_max, _min=_min,
                loss=engine.loss, dataset_name=dataset_name)
    print(teacher_model)

    # student_model
    # STMLP
    NAS = [1024, 2048]
    student_model = MLP_Student(depth=6, dropout2=0.1, num_layers=len(NAS) + 1, input_dim=64,
                                hidden_dim=128, output_dim=64, dropout_ratio=0.3, hidden_list=NAS,
                                output=12).to('cuda')
    # MLP
    # NAS = [256,512,1024,2048,1024,512,256]
    # NAS=[256,512,256]
    # student_model = MLP_Student(depth=6, dropout2=0.1, num_layers=len(NAS) + 1, input_dim=12,
    #                             hidden_dim=128, output_dim=12, dropout_ratio=0.3, hidden_list=NAS,
    #                             output=12).to('cuda')
    print(student_model)
    engine = trainer(scaler, teacher_model,student_model, **optim_args)
    engine.set_resume_lr_and_cl(resume_epoch, batch_num)
    # student_train(base_lr=0.01,
    #             patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
    #               test_every_n_epochs=10, epsilon=1e-8,)
    # =============================================================== Training ================================================================= #
    if mode != 'test':
        for epoch in range(resume_epoch + 1, optim_args['epochs']):
            # train a epoch
            time_train_start = time.time()

            current_learning_rate = engine.lr_scheduler.get_last_lr()[0]
            train_loss = []
            train_mape = []
            train_rmse = []
            dataloader['train_loader'].shuffle()  # traing data shuffle when starting a new epoch.
            for itera, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
                trainx = data_reshaper(x, device)
                trainy = data_reshaper(y, device)
                mae, mape, rmse = engine.train(trainx, trainy, batch_num=batch_num, _max=_max, _min=_min)
                print("{0}: {1}".format(itera, mae), end='\r')
                train_loss.append(mae)
                train_mape.append(mape)
                train_rmse.append(rmse)
                batch_num += 1
            time_train_end = time.time()
            train_time.append(time_train_end - time_train_start)

            current_learning_rate = engine.optimizer.param_groups[0]['lr']

            if engine.if_lr_scheduler:
                engine.lr_scheduler.step()
            # record history loss
            mtrain_loss = np.mean(train_loss)
            mtrain_mape = np.mean(train_mape)
            mtrain_rmse = np.mean(train_rmse)
            # =============================================================== Validation ================================================================= #
            time_val_start = time.time()
            mvalid_loss, mvalid_mape, mvalid_rmse, = engine.eval(device, dataloader, model_name, _max=_max, _min=_min)
            time_val_end = time.time()
            val_time.append(time_val_end - time_val_start)

            curr_time = str(time.strftime("%d-%H-%M", time.localtime()))
            log = 'Current Time: ' + curr_time + ' | Epoch: {:03d} | Train_Loss: {:.4f} | Train_MAPE: {:.4f} | Train_RMSE: {:.4f} | Valid_Loss: {:.4f} | Valid_RMSE: {:.4f} | Valid_MAPE: {:.4f} | LR: {:.6f}'
            print(log.format(epoch, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_rmse, mvalid_mape,
                             current_learning_rate))
            early_stopping(mvalid_loss, engine.model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
            # =============================================================== Test ================================================================= #
            engine.test_student(student_model, save_path_resume, device, dataloader, scaler, model_name, _max=_max, _min=_min,
                        loss=engine.loss, dataset_name=dataset_name)

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(val_time)))
    else:
        engine.test(model, save_path_resume, device, dataloader, scaler, model_name, save=False, _max=_max, _min=_min,
                    loss=engine.loss, dataset_name=dataset_name)


# def student_train(base_lr,
#                 patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
#                   test_every_n_epochs=10, epsilon=1e-8, **kwargs):
#     # steps is used in learning rate - will see if need to use it?
#     min_val_loss = float('inf')
#     wait = 0
#     optimizer = torch.optim.Adam(student_model.parameters(), lr=base_lr, eps=epsilon)
#
#     # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
#     #                                                     gamma=lr_decay_ratio)
#
#     # self._logger.info('Start training ...')
#
#     # this will fail if model is loaded with a changed batch_size
#     num_batches = 32
#     # self._logger.info("num_batches:{}".format(num_batches))
#     epoch_num=200
#     batches_seen = num_batches * epoch_num
#
#     for epoch_num in range(epoch_num, epochs):
#
#         student_model = student_model.train()
#
#         train_iterator = self._data['train_loader'].get_iterator()
#         losses = []
#         start_time = time.time()
#
#         for _, (x_old, y) in enumerate(train_iterator):
#             optimizer.zero_grad()
#
#             x, y = self._prepare_data(x_old, y)
#
#             teacher_output = self.dcrnn_model(x)
#             student_output = self.student_model(torch.tensor(x_old).cuda())
#             student_output = student_output.permute(3, 0, 2, 1)[:, :, :, 0]
#             # MSELoss = nn.MSELoss()
#             # loss=MSELoss(student_output,teacher_output)
#             loss = self._compute_mseloss(teacher_output, student_output)
#             self._logger.debug(loss.item())
#
#             losses.append(loss.item())
#
#             batches_seen += 1
#             loss.backward()
#
#             # gradient clipping - this does it in place
#             torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), self.max_grad_norm)
#
#             optimizer.step()
#
#         self._logger.info("epoch complete")
#         lr_scheduler.step()
#         self._logger.info("evaluating now!")
#
#         val_loss, _ = self.evaluate(dataset='val')
#
#         end_time = time.time()
#
#         self._writer.add_scalar('training loss',
#                                 np.mean(losses),
#                                 batches_seen)
#
#         if (epoch_num % log_every) == log_every - 1:
#             message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
#                       '{:.1f}s'.format(epoch_num, epochs, batches_seen,
#                                        np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
#                                        (end_time - start_time))
#             self._logger.info(message)
#
#         if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
#             test_loss, _ = self.evaluate4(dataset='test')
#             message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
#                       '{:.1f}s'.format(epoch_num, epochs, batches_seen,
#                                        np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
#                                        (end_time - start_time))
#             self._logger.info(message)
#
#         if val_loss < min_val_loss:
#             wait = 0
#             test_loss, _ = self.evaluate4(dataset='test')
#             message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
#                       '{:.1f}s'.format(epoch_num, epochs, batches_seen,
#                                        np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
#                                        (end_time - start_time))
#             self._logger.info(message)
#
#             if save_model:
#                 model_file_name = self.save_model(epoch_num)
#                 self._logger.info(
#                     'Val loss decrease from {:.4f} to {:.4f}, '
#                     'saving to {}'.format(min_val_loss, val_loss, model_file_name))
#             min_val_loss = val_loss
#
#         elif val_loss >= min_val_loss:
#             wait += 1
#             if wait == patience:
#                 self._logger.warning('Early stopping at epoch: %d' % epoch_num)
#                 break


if __name__ == '__main__':
    t_start = time.time()
    main()
    t_end = time.time()
    print("Total time spent: {0}".format(t_end - t_start))
