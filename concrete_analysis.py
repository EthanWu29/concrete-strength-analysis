import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def design_matrix(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return np.column_stack([np.ones(len(x)), x])


def evaluate(y, prediction):
    y = np.asarray(y).ravel()
    prediction = np.asarray(prediction).ravel()
    mse = float(np.mean((y - prediction) ** 2))
    return {'MSE': mse, 'R2': float(1 - mse / np.var(y, ddof=0))}


def gradient_descent(x, y, learning_rate=0.01,
                     max_iterations=20000, tolerance=1e-10):
    a = design_matrix(x)
    y = np.asarray(y, dtype=float).ravel()
    if len(y) != len(a) or not np.isfinite(a).all() or not np.isfinite(y).all():
        raise ValueError('check the input data')
    if learning_rate <= 0 or max_iterations < 1 or tolerance < 0:
        raise ValueError('check the training settings')
    theta = np.zeros(a.shape[1])
    history = [float(np.mean((a @ theta - y) ** 2))]
    converged = False
    for _ in range(max_iterations):
        residuals = a @ theta - y
        gradient = 2 / len(y) * (a.T @ residuals)
        new_theta = theta - learning_rate * gradient
        new_loss = float(np.mean((a @ new_theta - y) ** 2))
        if not np.isfinite(new_loss):
            raise FloatingPointError('try a smaller learning rate')
        previous_loss = history[-1]
        theta = new_theta
        history.append(new_loss)
        change = abs(previous_loss - new_loss) / max(1, previous_loss)
        if change <= tolerance:
            converged = True
            break
    return {'theta': theta, 'loss_history': np.array(history),
            'iterations': len(history) - 1, 'converged': converged}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Concrete_Data.csv')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = ['Cement', 'Slag', 'FlyAsh', 'Water', 'Superplasticizer',
             'CoarseAggregate', 'FineAggregate', 'Age', 'Strength']
    df = pd.read_csv(args.data)
    if df.shape != (1030, 9) or df.isna().any().any():
        raise ValueError('expected 1030 complete rows and 9 columns')
    df.columns = names
    # keep the original row order
    train = pd.concat([df.iloc[:500], df.iloc[630:]])
    test = df.iloc[500:630]
    x = train.iloc[:, :8].to_numpy(float)
    xt = test.iloc[:, :8].to_numpy(float)
    y = train.Strength.to_numpy(float)
    yt = test.Strength.to_numpy(float)
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    std = np.where(std == 0, 1, std)
    settings = {'Untransformed': (x, xt, 2e-7),
                'Standardized': ((x - mean) / std, (xt - mean) / std, .05),
                'Log-transformed': (np.log1p(x), np.log1p(xt), .003)}
    optimal_rows, gd_rows, pvalues, histories = [], [], {}, {}
    fitted = {}
    for name, (a, at, lr) in settings.items():
        gd = gradient_descent(a, y, lr)
        histories[name] = gd['loss_history']
        positive = np.flatnonzero(gd['loss_history'] < np.var(y, ddof=0))
        gd_rows.append({'Model': name, 'Learning rate': lr,
                        'Iterations': gd['iterations'], 'Converged': gd['converged'],
                        'Training MSE': gd['loss_history'][-1],
                        'First positive R2': int(positive[0]) if len(positive) else None})
        model = LinearRegression().fit(a, y)
        fitted[name] = model
        train_metrics, test_metrics = evaluate(y, model.predict(a)), evaluate(yt, model.predict(at))
        optimal_rows.append({'Model': name, 'Training MSE': train_metrics['MSE'],
                             'Testing MSE': test_metrics['MSE'],
                             'Training R2': train_metrics['R2'], 'Testing R2': test_metrics['R2']})
        pvalues[name] = np.asarray(sm.OLS(y, sm.add_constant(a)).fit().pvalues)[1:]
    optimal = pd.DataFrame(optimal_rows)
    optimal.to_csv(out / 'optimal_models.csv', index=False)
    pd.DataFrame(gd_rows).to_csv(out / 'gradient_descent.csv', index=False)
    pd.DataFrame(pvalues, index=names[:-1]).to_csv(out / 'pvalues.csv', index_label='Feature')
    uni_rows, coefficients = [], []
    for i, name in enumerate(names[:-1]):
        a = design_matrix(x[:, i])
        largest = np.linalg.eigvalsh(a.T @ a / len(y))[-1]
        lr = .9 / largest
        gd = gradient_descent(x[:, i], y, lr)
        positive = np.flatnonzero(gd['loss_history'] < np.var(y, ddof=0))
        metric = evaluate(y, design_matrix(x[:, i]) @ gd['theta'])
        uni_rows.append({'Feature': name, 'Learning rate': lr,
                         'Iterations': gd['iterations'], 'Training R2': metric['R2'],
                         'First positive R2': int(positive[0]) if len(positive) else None})
        uni = LinearRegression().fit(x[:, i:i+1], y)
        coefficients.append({'Feature': name, 'Univariate': uni.coef_[0],
                             'Multivariate': fitted['Untransformed'].coef_[i]})
    pd.DataFrame(uni_rows).to_csv(out / 'univariate_training.csv', index=False)
    pd.DataFrame(coefficients).to_csv(out / 'coefficients.csv', index=False)
    train.describe().T[['mean', 'std', 'min', 'max']].to_csv(out / 'training_summary.csv')
    # full and zoomed loss curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax in axes:
        for name, loss in histories.items():
            ax.plot(np.arange(len(loss)), loss, label=name, linewidth=1.4)
        ax.set_xlabel('Parameter updates')
        ax.set_ylabel('Training MSE (MPa²)')
        ax.grid(alpha=.2)
    axes[1].set_xlim(0, 5000)
    axes[1].set_ylim(50, 300)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'loss_comparison.png', dpi=200)
    plt.close(fig)
    versions = {'Python': platform.python_version(), 'NumPy': np.__version__,
                'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
                'scikit-learn': sklearn.__version__, 'statsmodels': statsmodels.__version__}
    (out / 'environment.json').write_text(json.dumps(versions, indent=2))
    print(optimal.to_string(index=False))
    print(pd.DataFrame(gd_rows).to_string(index=False))
    print(pd.DataFrame(uni_rows).to_string(index=False))
    print('Environment:', versions)


if __name__ == '__main__':
    main()
