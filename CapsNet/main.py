import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
from model import CapsuleNet
from layers import caps_loss
from utils import load_mnist, show_reconstruction


def test(model, test_loader, args):
    """return: test loss, test accuracy

    test_loader: torch.utils.data.DataLoader for test data
    """
    model.eval()
    test_loss = 0
    correct = 0

    for x, y in test_loader:
        y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x.cuda(), volatile=True), Variable(y.cuda())

        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).data.item() * x.size(0)  # sum up batch loss

        y_pred_ = y_pred.max(1, keepdim=True)[1]
        y_true_ = y.max(1, keepdim=True)[1]
        correct += y_pred_.eq(y_true_).sum().item()

    test_loss /= len(test_loader.dataset)

    return test_loss, correct / len(test_loader.dataset)


def train(model, train_loader, test_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)

    from time import time
    import csv

    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc = 0.

    for epoch in range(args.epochs):
        model.train()
        lr_decay.step()
        ti = time()
        training_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()
            y_pred, x_recon = model(x, y)
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)
            loss.backward()
            training_loss += loss.data.item() * x.size(0)
            optimizer.step()

            if i % 100 == 0:
                val_loss, val_acc = test(model, test_loader, args)
                print("==> Epoch %02d, step %06d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
                    % (epoch, i, training_loss / len(train_loader.dataset),
                        val_loss, val_acc, time() - ti))

        # compute validation loss and acc
        val_loss, val_acc = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))

        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)

    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)

    return model


def load_config():
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.0005 * 784, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixel6s to shift at most in each direction.")
    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")
    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    return args

if __name__ == '__main__':
    args = load_config()

    # load data
    train_loader, test_loader = load_mnist(args.data_dir, download=False, batch_size=args.batch_size)

    # define model
    model = CapsuleNet(input_size=[1, 28, 28], classes=10, iterations=3)
    model.cuda()
    print(model)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))

    if not args.testing:
        train(model, train_loader, test_loader, args)

    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')

        test_loss, test_acc = test(model=model, test_loader=test_loader, args=args)
        print('test acc = %.4f, test loss = %.5f' % (test_acc, test_loss))
        show_reconstruction(model, test_loader, 50, args)



