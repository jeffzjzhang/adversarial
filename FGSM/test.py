import torch.nn.functional as F

from data import *
from fgsm import *

MAX_ADV_EX_NUM = 10

# Test stuff
def test(model, device, test_loader, epsilon):
    correct_counter = 0
    adversarial_examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        if init_pred.item() != target.item():
            continue

        loss = F.nll_loss(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output_again = model(perturbed_data)
        final_pred = output_again.max(1, keepdim=True)[1]

        if final_pred.item() == target.item():
            correct_counter += 1
        else:
            if len(adversarial_examples) < MAX_ADV_EX_NUM:
                adversarial_example = perturbed_data.squeeze().detach().cpu().numpy()
                adversarial_examples.append(adversarial_example)
        
    final_accuracy = correct_counter / float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy: {} / {} = {}".format(epsilon, correct_counter, len(test_loader), final_accuracy))

    return final_accuracy, adversarial_examples
