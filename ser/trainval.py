import torch
import torch.nn.functional as F
import os


def train(
    main_model,
    epochs,
    training_dataloader,
    validation_dataloader,
    device,
    optimizer,
    directory,
):
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(training_dataloader):
            images, labels = images.to(device), labels.to(device)
            main_model.train()
            optimizer.zero_grad()
            output = main_model(images)
            loss = F.nll_loss(output, labels)
            loss.backward()
            optimizer.step()
            print(
                f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
                f"| Loss: {loss.item():.4f}"
            )
        validate(main_model, validation_dataloader, device, epoch, directory)


def validate(main_model, validation_dataloader, device, epoch, directory):
    best_scores = {"accuracy": 0, "epoch": 0}
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            main_model.eval()
            output = main_model(images)
            val_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(validation_dataloader.dataset)
        val_acc = correct / len(validation_dataloader.dataset)

        print(f"Validation: Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}")

        if val_acc >= best_scores["accuracy"]:
            best_scores["accuracy"] = val_acc
            best_scores["epoch"] = epoch
            torch.save(
                main_model.cpu().state_dict(), os.path.join(directory, "model.pt")
            )

    return
