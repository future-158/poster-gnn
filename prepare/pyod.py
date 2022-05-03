import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
import torch
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
from pygod.models import DOMINANT
from pygod.utils.metric import \
    eval_roc_auc, \
    eval_recall_at_k, \
    eval_precision_at_k

# load dataset
# foreach snapshot in snapshos
    # model = DOMINANT()
    # model.fit(data)
    # labels = model.predict(data)

