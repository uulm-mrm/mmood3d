import warnings

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import Tensor


def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        #image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_hwc
        #return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image

def plot_features_ood(obj_feats: Tensor, gt_labels_3d):
    obj_feats_np = obj_feats.cpu().detach().numpy()
    gt_labels_3d_np = gt_labels_3d.cpu().detach().numpy()

    perplexity = min(50, len(obj_feats_np) // 3)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, learning_rate="auto", init="pca")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tsne_result = tsne.fit_transform(obj_feats_np)

    tsne_id_feats = tsne_result[gt_labels_3d_np == 0]
    tsne_ood_feats = tsne_result[gt_labels_3d_np == 1]

    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_id_feats[:, 0], tsne_id_feats[:, 1], label="ID Features", color='green', alpha=0.5, s=50,
                marker='o')
    plt.scatter(tsne_ood_feats[:, 0], tsne_ood_feats[:, 1], label="OOD Features", color='red', alpha=0.5, s=50,
                marker='s')

    plt.legend()
    plt.title("Object Features")

    return plt.gcf()
