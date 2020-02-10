import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, plt, line_color, axes=None, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(24, 6))
        axes[0].grid()
        axes[1].grid()
        axes[2].grid()
        axes[0].set_ylim(0.85, 1.01)
        axes[0].set_title('Learning Curves')

        if ylim is not None:
            axes[0].set_ylim(*ylim)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    # fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    # axes[0].grid()

    # axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-',
                 label="Train " + title, linewidth=2, color=line_color)
    axes[0].plot(train_sizes, test_scores_mean, 'o-',
                 label="CV " + title, linewidth=5, color=line_color)
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    # axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-', label=title, color=line_color)
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                      fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of models")
    axes[1].legend(loc="best")

    # Plot fit_time vs score
    # axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-', label=title, color=line_color)
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of model")
    axes[2].legend(loc="best")
    # plt.savefig(title + '_learning_curve_scalability_performance' + '.png')
    # plt.close()
    return plt, axes


def plot_validation_curve(estimator, title, X, y, param_name, param_range, best_val, cv=None):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, cv=cv,
        scoring="accuracy", n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # best_vals = (test_scores_mean + test_scores_std) / (test_scores_mean - test_scores_std)
    best_val_index = np.argmax(test_scores_mean)
    best_val_new = param_range[best_val_index]
    print('best_val=' + str(best_val))
    print('best_val_new=' + str(best_val_new))

    _, ax = plt.subplots(1, 1,  figsize=(7, 5) )
    ax.set_xscale('linear')
    plt.title("Validation Curve for " + title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    # plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")

    # plt.xticks(xticklabels)
    # ax.xaxis.set_major_locator(ticker.AutoLocator())
    # ax.xaxis.set_minor_locator(ticker.NullLocator())
    # plt.xticks(range(min(param_range), max(param_range) + 1))
    trans = transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    # plt.axvline(x=best_val, dashes=(5, 3))
    # plt.text(best_val, 0.7, ' best\n ' + param_name + '\n = ' + str(best_val), transform=trans)
    plt.axvline(x=best_val_new, dashes=(5, 3))
    plt.text(best_val_new, 0.5, ' best\n ' + param_name + '\n = ' + str(best_val_new), transform=trans)
    # ax.set_xticklabels(xticklabels)
    plt.savefig(title + '_validation' + '.png')
    plt.close()