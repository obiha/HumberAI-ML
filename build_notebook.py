"""Build and execute notebooks/analysis.ipynb, saved with its outputs.

This script is a one-off generator, not part of the latedeliveries package.
Run it with the lab05-late-deliveries environment active.
"""

import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("# Late Deliveries: analysis\n\nHow common late orders are, how the "
   "candidate models compare, how the threshold was chosen from the "
   "scenario's costs, and the honest test-set result.")

code(
    "import sys\n"
    "sys.path.insert(0, '../src')\n"
    "import json\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "from sklearn.metrics import ConfusionMatrixDisplay\n"
    "\n"
    "from latedeliveries.data import make_orders\n"
    "from latedeliveries.pipeline import split_data\n"
    "\n"
    "frame = make_orders()\n"
    "frame.head()"
)

md("## How common are late orders?")

code(
    "late_rate = frame['late'].mean()\n"
    "print(f'Late rate: {late_rate:.1%} ({frame[\"late\"].sum()} of "
    "{len(frame)} orders)')\n"
    "frame['late'].value_counts(normalize=True).rename({0: 'on time', "
    "1: 'late'}).plot(kind='bar', title='Share of orders by outcome')\n"
    "plt.ylabel('share of orders')\n"
    "plt.show()"
)

code(
    "frame.isna().mean().loc[lambda s: s > 0].rename('share missing')"
)

md(
    "About one order in six arrives late, matching the scenario. "
    "`prep_minutes` and `customer_rating` have missing values, which is one "
    "more reason `customer_rating` (missing for half the orders, and only "
    "recorded after delivery) is unusable as a feature."
)

md("## Comparing models\n\nFive-fold stratified cross-validation on the "
   "training data only (see `src/latedeliveries/train.py`). Results are "
   "also written to `reports/cv_results.csv`.")

code(
    "cv_results = pd.read_csv('../reports/cv_results.csv')\n"
    "cv_results"
)

md(
    "`logistic_regression_balanced` was chosen: its ROC-AUC "
    "(ability to rank orders by late risk) is essentially tied with plain "
    "logistic regression and clearly ahead of the random forest, and its "
    "recall is far higher (about 0.67 vs 0.24), which matters because "
    "missing a late order costs $15 while flagging one costs only $4. "
    "`GridSearchCV` over `C` and `solver` then barely moved ROC-AUC "
    "(0.7889 tuned vs 0.7881 +/- 0.0368 untuned), well inside the "
    "fold-to-fold spread, so tuning did not meaningfully beat the default "
    "settings here."
)

md("## Choosing the threshold from costs\n\nOut-of-fold probabilities "
   "from the training data only, scored against the scenario's cost rule "
   "for every threshold from 0.05 to 0.95.")

code(
    "with open('../reports/metrics.json') as f:\n"
    "    metrics = json.load(f)\n"
    "\n"
    "cost_table = pd.DataFrame(metrics['cost_table'])\n"
    "plt.plot(cost_table['threshold'], cost_table['cost'], marker='o')\n"
    "plt.axvline(metrics['threshold'], color='red', linestyle='--', "
    "label=f\"chosen threshold = {metrics['threshold']}\")\n"
    "plt.xlabel('threshold')\n"
    "plt.ylabel('total cost on training folds ($)')\n"
    "plt.title('Cost against threshold (out-of-fold, training data only)')\n"
    "plt.legend()\n"
    "plt.show()"
)

md("## Test-set result (scored once)")

code(
    "cm = metrics['test_metrics']['confusion_matrix']\n"
    "ConfusionMatrixDisplay(confusion_matrix=pd.DataFrame(cm).values,\n"
    "                       display_labels=['on time', 'late']).plot()\n"
    "plt.title('Test-set confusion matrix at the chosen threshold')\n"
    "plt.show()\n"
    "metrics['test_metrics']"
)

code(
    "policy_costs = pd.Series(metrics['policy_costs'])\n"
    "policy_costs.plot(kind='bar', title='Total test-set cost by policy')\n"
    "plt.ylabel('cost ($)')\n"
    "plt.show()\n"
    "policy_costs"
)

md(
    "## Recommendation\n\n"
    "**Would you use this model?** Yes. Flagging with the model costs "
    "$1,481 on the test set, versus $1,755 for flagging nobody (today's "
    "policy) and $3,000 for flagging everybody - a real reduction with no "
    "change in company process beyond sending a voucher.\n\n"
    "**Why this threshold?** 0.6 was the point on the training-only, "
    "out-of-fold cost curve that minimised the scenario's cost "
    "($4 per voucher, $15 per missed late order); it is not the default "
    "0.5 cut-off and was never chosen by looking at the test set.\n\n"
    "**What did you exclude and why?** `order_id` (an identifier), "
    "`prep_minutes` (not known until the kitchen finishes, after the order "
    "is placed) and `customer_rating` (missing for half the orders and "
    "only given after the order has already arrived; in this simulated "
    "data it is even partly derived from `late` itself). Adding "
    "`prep_minutes` and `customer_rating` back lifts CV ROC-AUC from 0.79 "
    "to about 0.92 - a sharp, suspicious jump that confirms they leak the "
    "outcome.\n\n"
    "**One limitation:** recall is about 0.43 at this threshold, so more "
    "than half of late orders still go unflagged; a model with a cheaper "
    "false-negative cost, or more features known at order time (e.g. "
    "current kitchen backlog), would likely do better."
)

nb["cells"] = cells

client = NotebookClient(nb, timeout=120, kernel_name="python3")
client.execute()

with open("notebooks/analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("notebook written")
