from sklearn.decomposition import PCA


def pca(X, n_components):
    # 创建pca模型实例，主成分个数为n_components个
    pca_model = PCA(n_components)
    # 模型拟合
    pca_model.fit(X)
    # 拟合模型并将模型应用于数据X
    x_trans = pca_model.transform(X)
    return x_trans

