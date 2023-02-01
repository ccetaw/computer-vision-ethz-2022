import torch
import time
from tqdm import tqdm
import argparse
from loader.dataloader_cifar10 import DataloaderCifar10
from models.vgg_simplified import Vgg
from utils import *
import torch.nn.functional as F


####### training settings #########
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--log_step", default=100, type=int, help='how many steps to log once')
parser.add_argument("--val_step", default=100, type=int)
parser.add_argument("--num_epoch", default=50, type=int, help='maximum num of training epochs')


parser.add_argument("--fc_layer", default=512, type=int, help='feature number the first linear layer in VGG')
parser.add_argument("--lr", default=0.0001, type=float, help='learning rate')

parser.add_argument("--save_dir", default='runs', type=str, help='path to save trained models and logs')
parser.add_argument("--root", default='data/data_cnn/cifar-10-batches-py', type=str, help='path to dataset folder')
args = parser.parse_args()
###################################


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# devide = torch.device('cpu')




def train(writer, logger):
    train_dataset = DataloaderCifar10(img_size=32, is_transform=True, split='train')
    train_dataset.load_data(args.root)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=2)

    val_dataset = DataloaderCifar10(img_size=32, is_transform=False, split='val')
    val_dataset.load_data(args.root)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=2)

    model = Vgg(fc_layer=args.fc_layer, classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # time_meter = averageMeter()

    total_steps = 0
    best_acc = 0
    for epoch in range(args.num_epoch):
        for step, data in tqdm(enumerate(train_dataloader)):
            start_ts = time.time()
            total_steps += 1

            model.train()
            imgs = data[0].to(device)  # [batch_size, 3, 32, 32]
            label = data[1].to(device)  # [batch_size]
            optimizer.zero_grad()
            preds = model(imgs)

            loss = F.cross_entropy(input=preds, target=label, reduction='mean')
            loss.backward()  # backpropagation loss
            optimizer.step()
            # time_meter.update(time.time() - start_ts)

            if total_steps % args.log_step == 0:
                print_str = '[Step {:d}/ Epoch {:d}]  Loss: {:.4f}  '.format(step, epoch, loss.item())
                logger.info(print_str)
                writer.add_scalar('train/loss', loss.item(), total_steps)
                print(print_str)
                # time_meter.reset()

            if total_steps % args.val_step == 0:
                model.eval()
                total, correct = 0, 0
                with torch.no_grad():
                    for step_val, data in tqdm(enumerate(val_dataloader)):
                        imgs = data[0].to(device)
                        labels = data[1].to(device)
                        preds = model(imgs)

                        _, predicted = preds.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                    acc = 100.0 * correct / total
                    logger.info('val acc:{}'.format(acc))
                    writer.add_scalar('val/acc', acc, total_steps)
                    print('acc:', acc)

                    if acc > best_acc:
                        best_acc = acc
                        state = {
                                "epoch": epoch,
                                "total_steps": total_steps,
                                "model_state": model.state_dict(),
                                "best_acc": acc,
                                }
                        save_path = os.path.join(writer.file_writer.get_logdir(), "last_model.pkl")
                        torch.save(state, save_path)
                        print('[*] best model saved\n')
                        logger.info('[*] best model saved\n')

        if epoch == args.num_epoch:
            break


if __name__ == '__main__':
    run_id = random.randint(1, 100000)
    logdir = os.path.join(args.save_dir, str(run_id))  # create new path
    writer = SummaryWriter(log_dir=logdir)
    print('RUNDIR: {}'.format(logdir))

    logger = get_logger(logdir)
    logger.info('Let the games begin')  # write in log file
    save_config(logdir, args)
    train(writer, logger)











