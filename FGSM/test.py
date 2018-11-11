import torch.nn.functional as F

from data import *
from fgsm import *
from model import *

MAX_ADV_EX_NUM = 10
TEST_SIZE = 2000

# Test stuff
def test(model, test_loader):
    model.eval()
    correct_num = 0
    total_loss = 0
    iterations = 0
    total_samples = 0

    for data, label in test_loader:
        if iterations > TEST_SIZE:
            break;
        data, label = data.to(device), label.to(device)
        output = model(data)
        _, preds = torch.max(output, 1)
        loss = nn.CrossEntropyLoss()(output, label)

        total_loss += loss.item() * data.size(0)
        correct_num += torch.sum(preds == label.data)

        iterations += 1
        total_samples += len(data)
    
    # avg_loss = total_loss / len(test_loader.dataset)
    # accuracy = correct_num.double() / len(test_loader.dataset)
    avg_loss = total_loss / total_samples
    accuracy = correct_num.double() / total_samples
    print("#" * 45)
    print('Test\tLoss: {:.4f}\tAccuracy: {:.4f}'.format(avg_loss, accuracy))
    print("#" * 45)


def attack(model, test_loader, epsilon):
    model.eval()
    correctly_defended = 0
    adversarial_examples = []
    iterations = 0
    total_samples = 0

    for data, target in test_loader:
        if iterations > TEST_SIZE:
            break
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = F.cross_entropy(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output_again = model(perturbed_data)
        final_pred = output_again.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correctly_defended += 1
        else:
            if len(adversarial_examples) < MAX_ADV_EX_NUM:
                adversarial_example = perturbed_data.squeeze().detach().cpu().numpy()
                adversarial_examples.append(adversarial_example)

        iterations += 1
        total_samples += len(data)
        
    # final_accuracy = correctly_defended / float(len(test_loader))
    final_accuracy = correctly_defended / float(total_samples)
    print("Epsilon: {}\tAccuracy after attack: {} / {} = {}".format(epsilon, correctly_defended, total_samples, final_accuracy))

    return final_accuracy, adversarial_examples
