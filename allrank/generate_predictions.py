from argparse import ArgumentParser, Namespace
from attr import asdict

from allrank.config import Config
from allrank.data.dataset_loading import create_data_loaders, load_libsvm_dataset_role
from allrank.data.dataset_loading import create_data_loaders
from allrank.inference.inference_utils import __rank_slates
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device, load_state_dict_from_file
from allrank.training.train_utils import compute_metrics

def parse_args() -> Namespace:
    parser = ArgumentParser("Generate predictions script")

    parser.add_argument("--ds_path", help="location of the dataset", required=True, type=str)

    parser.add_argument("--model_path", help="location of saved model", required=True, type=str)

    parser.add_argument("--output_path", help="generated predictions will be saved to this path", required=True, type=str)

    parser.add_argument("--slate_length", help="all slates will be padded to this value for evaluation", required=True, type=int)

    parser.add_argument("--role", help="role of the dataset to be loaded", required=False, type=str, default="test")

    return parser.parse_args()

args = parse_args()

dev = get_torch_device()

prediction_ds = load_libsvm_dataset_role(args.role, args.ds_path, args.slate_length)
_, prediction_dl = create_data_loaders(prediction_ds, prediction_ds, num_workers=1, batch_size=prediction_ds.shape[0])

config = Config.from_json(f"{args.model_path}/used_config.json")
model = make_model(n_features=prediction_ds.shape[-1], **asdict(config.model, recurse=False))

model.load_state_dict(load_state_dict_from_file(f"{args.model_path}/model.pkl", dev))

X, y = __rank_slates(prediction_dl, model)

with open(f"{args.output_path}/{args.role}_predictions.csv", "w") as file:
    file.write(f"qid,eid,pred_score,true_sorted_by_pred\n")
    for i, (qid, count) in enumerate(prediction_dl.dataset.query_ids.items()):
        for j in range(count):
            file.write(f"{qid},{int(X[i, j, -1])},{args.slate_length - j},{int(y[i, j])}\n")

result_metrics = compute_metrics(config.metrics, model, prediction_dl, dev)
for metric_name, metric_value in result_metrics.items():
    print(f"{metric_name}: {metric_value}")