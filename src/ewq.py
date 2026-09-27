#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import torch
import torch.nn.functional as F
import onnx
from onnx import numpy_helper
import numpy as np

def _weight_entropy(weight_matrix, epsilon=0.01):
    w_flat = weight_matrix.view(-1).float()
    size_w = w_flat.numel()
    p = F.softmax(w_flat, dim=0)
    entropy = -torch.sum(p * torch.log(p + epsilon)).item()
    return entropy, size_w

def _block_ewq_score(weight_list, epsilon=0.01):
    total_weighted = 0.0
    total_params = 0

    for W in weight_list:
        if isinstance(W, torch.nn.Parameter):
            W = W.data
        h_wi, size_wi = _weight_entropy(W, epsilon)
        total_weighted += h_wi / size_wi
        total_params += size_wi

    if total_params > 0:
        return total_weighted / total_params
    return 0.0


# In[2]:


def _normalize_scores(scores):
    arr = np.array(scores, dtype=np.float64)
    min_val = arr.min()
    max_val = arr.max()
    normalized = (arr - min_val) / (max_val + 1e-8)
    return normalized.tolist()


def _rank_layers(scores):
    arr = np.array(scores)
    ranked = np.argsort(arr).tolist()
    return ranked


def _build_output(metric_name, model_path, scores, extra=None):
    model_name = os.path.basename(model_path.rstrip("/\\"))
    num_layers = len(scores)
    layer_labels = [f"layer_{i}" for i in range(1, num_layers + 1)]
    layer_map = {label: float(s) for label, s in zip(layer_labels, scores)}
    out = {
        "metric": metric_name,
        "model": model_name,
        "num_layers": num_layers,
        "layer_scores": layer_map,
        "ranked_layers": _rank_layers(scores),
        "scores": scores,
    }
    if extra:
        out.update(extra)
    return out

def _get_sensitive_nodes():
    nodes = []

    return nodes

def _save_output(output_dict, output_dir, metric_name, model_path):
    model_name = os.path.basename(model_path.rstrip("/\\"))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{model_name}_{metric_name}.json")
    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=2)
    return out_path


# In[3]:


class ewq():
	def __init__(self, model_path="yolo26n-seg.onnx", target=0.35):
		self.model_path = model_path
		self.target = target

	def filter(self):
		threshold = 1.0
		layer_scores = []
		layer_names = []

		model = onnx.load(self.model_path)
		inits = {init.name: init for init in model.graph.initializer}  # Get the initializers of the original modelq
		for init in model.graph.initializer:
			if "weight" in init.name and "conv" in init.name:
				# print("\n",init.name)
				scores = []

				layer_names.append(init.name)
				total_params = 0
				W_np = numpy_helper.to_array(inits[init.name])
				weights = torch.from_numpy(W_np).float()
				score = _block_ewq_score([weights])
				# print(score)
				layer_scores.append(score)
				# print(layer_scores)

		output_dir = "./results"
		model_name = os.path.basename(self.model_path.rstrip("/"))
		os.makedirs(output_dir, exist_ok=True)
		out_file = os.path.join(output_dir, f"{model_name}_ewq.json")
		result = {"scores": layer_scores, "layer_names": layer_names}
		with open(out_file, "w") as f:
			json.dump(result, f, indent=2)

	    ### calculate scores
		raw_scores = result.get("scores", [])
		layer_names = result.get("layer_names", [])
		# print(raw_scores)
		# print(layer_names)
		raw_scores = _normalize_scores(raw_scores)
		# print(raw_scores)

		ranked_nodes = []

		print(f"\n[*] Top-5 most sensitive layers (0-indexed):")
		ranked = _rank_layers(raw_scores)
		for rank, layer_idx in enumerate(ranked[:5]):
			print(f"    Rank {rank + 1}: Layer {layer_names[layer_idx].replace('.weight', '')}  (score={raw_scores[layer_idx]:.6f})")

		for rank, layer_idx in enumerate(ranked):
			ranked_nodes.append(layer_names[layer_idx].replace('.weight', ''))

		#### helper func
		weight_params = []
		total_param = sum(int(np.prod(init.dims)) for init in model.graph.initializer)
		target_param = (1 - self.target) / 0.75 * total_param

		inits = {init.name: init for init in model.graph.initializer if "weight" in init.name and "conv" in init.name}
		init_to_node = {}
		for node in model.graph.node:
			### init_to_node ref
			init_to_node.update({inp: node.name for inp in node.input if inp in inits})
			### store params per-node
			n_params = sum(int(np.prod(inits[inp].dims)) for inp in node.input if inp in inits)
			if n_params > 0:
				weight_params.append((node.name, n_params))
		#### end helper func

		selected, acc = [], 0
		for i in range(len(weight_params )):
			name, w = weight_params[i]
			selected.append(name)
			acc += w
			if acc >= target_param:
				acc -= w
				break

		# print(acc)
		# print(target_param)
		# print(len(ranked_nodes))
		# print(len(selected))
		selected
		print(len(selected))
		print(acc)
		return selected


# In[4]:


if __name__ == '__main__':
	ewq = ewq()
	ewq.filter()	


# In[ ]:




