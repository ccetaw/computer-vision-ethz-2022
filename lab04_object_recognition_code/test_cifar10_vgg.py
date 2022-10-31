import torch
from tqdm import tqdm
import argparse
from loader.dataloader_cifar10 import DataloaderCifar10
from models.vgg_simplified import Vgg


####### training settings #########
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)


parser.add_argument("--fc_layer", default=512, type=int)

parser.add_argument("--model_path", default='runs/48557/last_model.pkl', type=str)
parser.add_argument("--root", default='data/data_cnn/cifar-10-batches-py', type=str, help='path to dataset folder')
args = parser.parse_args()
###################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# devide = torch.device('cpu')




def test(args):
    test_dataset = DataloaderCifar10(img_size=32, is_transform=True, split='test')
    test_dataset.load_data(args.root)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=0)

    model = Vgg(fc_layer=args.fc_layer, classes=10).to(device)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()


    total, correct = 0, 0
    with torch.no_grad():
        for step, data in tqdm(enumerate(test_dataloader)):
            imgs = data[0].to(device)  # [batch_size, 3, 32, 32]
            labels = data[1].to(device)  # [batch_size]
            preds = model(imgs)

            _, predicted = preds.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print('test accuracy:', acc)




if __name__ == '__main__':
    test(args)











