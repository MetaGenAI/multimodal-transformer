import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(os.path.join(THIS_DIR, os.pardir), os.pardir))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
EXTRACT_DIR = os.path.join(DATA_DIR, 'extracted_data')
if not os.path.isdir(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.isdir(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

sys.path.append(ROOT_DIR)

import time
from scripts.training.data import create_dataset, create_dataloader
from models import create_model
import random
from scripts.training.options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()
    model = create_model(opt)
    #print("HI")
    model.setup()
    train_dataset = create_dataset(opt)
    train_dataset.setup()
    train_dataloader = create_dataloader(train_dataset)
    if opt.val_epoch_freq:
        val_dataset = create_dataset(opt, validation_phase=True)
        val_dataset.setup()
        val_dataloader = create_dataloader(val_dataset)
    print('#training sequences = {:d}'.format(len(train_dataset)))

    total_steps = 0

    for epoch in range(opt.epoch_count, opt.nepoch + opt.nepoch_decay):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(train_dataloader):
            # print(i)
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            if total_steps % opt.display_freq == 0 or total_steps % opt.print_freq == 0:
                model.evaluate_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                print(losses)
                metrics = model.get_current_metrics()
                print(metrics)
                t = (time.time() - iter_start_time) / opt.batch_size

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

                iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.nepoch + opt.nepoch_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

        if opt.val_epoch_freq and epoch % opt.val_epoch_freq == 0:
            val_start_time = time.time()
            with model.start_validation() as update_validation_meters:
                if opt.eval:
                    model.eval()
                for j, data in enumerate(val_dataloader):
                    val_start_time = time.time()
                    model.set_input(data)
                    model.test()
                    model.evaluate_parameters()
                    update_validation_meters()
            losses_val = model.get_current_losses(is_val=True)
            metrics_val = model.get_current_metrics(is_val=True)
            print("Validated parameters at epoch {:d} \t Time Taken: {:d} sec".format(epoch, int(time.time() - val_start_time)))
