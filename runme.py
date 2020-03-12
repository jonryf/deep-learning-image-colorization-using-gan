from pix2pix import pix2pix

def TenToPic(image):
    s = image.size()
    ret = torch.zeros(s[1], s[2], s[0])
    for i in range(s[0]):
        ret[:, :, i] = image[i, :,:]
    return ret.detach().numpy().astype(int)


model = pix2pix().cuda()
trainPix2Pix(model, train_dataset, totalEpochs=200, genLr=0.001, descLr=0.001)