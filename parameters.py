batch_size = 100
num_classes = 10
img_rows, img_cols = 28, 28


epochs = [5]

conv2d_params = {
    'filters': [16, 32, 64, 128],
    'kernel_size': [(3,3), (5,5)],
    'padding': ['valid'],
    'activation': ['relu', 'sigmoid']
}

pooling_params = {
    'pool_size': [(2,2), (4,4)]
}
