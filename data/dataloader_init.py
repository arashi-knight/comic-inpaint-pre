from data import comic_dataloader


def get_dataloader(config):
    """返回classes, trainloader, testloader"""
    if config.dataset_name == 'comic':
        x = comic_dataloader.Dataloder(config)
        return x.getloader()