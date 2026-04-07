from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=True)

print("下载完成：", path)
