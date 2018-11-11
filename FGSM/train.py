from model import *

def train(model, train_loader, optimizer, num_epochs=20):
    for epoch in range(num_epochs):
        print("Epoch {} / {}".format(epoch, num_epochs - 1))

        model.train()
        running_loss = 0.0
        correct_num = 0
        iterations = 0
        total_samples = 0
        print(len(train_loader))
        for data, label in train_loader:
            if iterations > 200:
                break
            data = data.to(device)
            label = label.to(device)
            if iterations % 20 == 0:
                print("Iter: %d" % iterations)
            optimizer.zero_grad()

            output = model(data)
            loss = nn.CrossEntropyLoss()(output, label)

            _, preds = torch.max(output, 1)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * data.size(0)
            correct_num += torch.sum(preds == label.data)

            iterations += 1
            total_samples += len(data)

        # epoch_loss = running_loss / len(train_loader.dataset)
        # epoch_accuracy = correct_num.double() / len(train_loader.dataset)
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_num.double() / total_samples
        print('Train\tLoss: {:.4f}\tAccuracy: {:.4f}'.format(epoch_loss, epoch_accuracy)) 
        
    return model
