import matplotlib.pyplot as plt
import numpy as np
import os
import json

def save_training_metrics(training_metrics, filename="training_metrics.json"):
  path = "training_metrics/" + filename
  os.makedirs(os.path.dirname(path), exist_ok=True)
  with open(path, "w", ) as f:
    json.dump(training_metrics, f)


def load_training_metrics(filename):
    path = "./" + filename
    with open(path, "r") as f:
        training_metrics = json.load(f)
    return training_metrics

# The output of load_training_metrics looks like the following:
# {'training_metrics': [{'epoch': 1, 'training_loss': 0.9195071952683586, 'training_accuracy': 0.7585034370422363, 'validation_loss': 0.3373750504106283, 'validation_accuracy': 0.8965986371040344}, {'epoch': 2, 'training_loss': 0.16449964301926748, 'training_accuracy': 0.9503401517868042, 'validation_loss': 0.27052116331954795, 'validation_accuracy': 0.9142857193946838}, {'epoch': 3, 'training_loss': 0.09533538488405091, 'training_accuracy': 0.9752834439277649, 'validation_loss': 0.29104752217729885, 'validation_accuracy': 0.9142857193946838}, {'epoch': 4, 'training_loss': 0.07378497911351067, 'training_accuracy': 0.981632649898529, 'validation_loss': 0.259120134015878, 'validation_accuracy': 0.91700679063797}, {'epoch': 5, 'training_loss': 0.054701253984655654, 'training_accuracy': 0.9888889193534851, 'validation_loss': 0.23712879108885923, 'validation_accuracy': 0.9238095283508301}, {'epoch': 6, 'training_loss': 0.04870404036981719, 'training_accuracy': 0.9900227189064026, 'validation_loss': 0.25535260296116274, 'validation_accuracy': 0.9238095283508301}, {'epoch': 7, 'training_loss': 0.0371109664440155, 'training_accuracy': 0.9927437901496887, 'validation_loss': 0.25412234105169773, 'validation_accuracy': 0.9265305995941162}, {'epoch': 8, 'training_loss': 0.04043357875198126, 'training_accuracy': 0.9900227189064026, 'validation_loss': 0.24353655396650234, 'validation_accuracy': 0.9142857193946838}, {'epoch': 9, 'training_loss': 0.029800518894834177, 'training_accuracy': 0.9945578575134277, 'validation_loss': 0.2514134996260206, 'validation_accuracy': 0.91700679063797}, {'epoch': 10, 'training_loss': 0.028406974433788233, 'training_accuracy': 0.9961451292037964, 'validation_loss': 0.24867038584003845, 'validation_accuracy': 0.9142857193946838}, {'epoch': 11, 'training_loss': 0.02752300248082195, 'training_accuracy': 0.9959183931350708, 'validation_loss': 0.24723887257277966, 'validation_accuracy': 0.9238095283508301}, {'epoch': 12, 'training_loss': 0.02813921786312546, 'training_accuracy': 0.9950113296508789, 'validation_loss': 0.2384933785845836, 'validation_accuracy': 0.9251700639724731}, {'epoch': 13, 'training_loss': 0.023610445910266467, 'training_accuracy': 0.9972789287567139, 'validation_loss': 0.2552756676450372, 'validation_accuracy': 0.9156462550163269}, {'epoch': 14, 'training_loss': 0.023434294281261307, 'training_accuracy': 0.9954648613929749, 'validation_loss': 0.24257866013795137, 'validation_accuracy': 0.922448992729187}, {'epoch': 15, 'training_loss': 0.020687680505216122, 'training_accuracy': 0.9959183931350708, 'validation_loss': 0.23906387388706207, 'validation_accuracy': 0.9238095283508301}], 'test_accuracy': 0.9156079888343811}

def plot_graph(training_metrics, type="loss", title="Loss"):
    plt.clf()
    plt.plot([x['epoch'] for x in training_metrics], [x['training_' + type] for x in training_metrics], label="Training")
    plt.plot([x['epoch'] for x in training_metrics], [x['validation_' + type] for x in training_metrics], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(type)
    plt.title(title)
    plt.legend()
    plt.savefig("./plots/" + title + ".png")

# Folders in training_metrics are num_layer_tests, transform_tests, and scheduler_tests

color_jitter_30 = load_training_metrics("results_30_test/training_metrics/transform_tests/color_jitter.json")
plot_graph(color_jitter_30['training_metrics'], title="Color Jitter 30%")

no_transform_30 = load_training_metrics("results_30_test/training_metrics/transform_tests/no_transform.json")
plot_graph(no_transform_30['training_metrics'], title="No Transform 30%")

incl_normalization_3 = load_training_metrics("results_30_test/training_metrics/num_layer_tests/3incl_normalization.json")
plot_graph(incl_normalization_3['training_metrics'], title="3 Layers with Normalization")

# All transform test in one graph
plt.clf()
color_jitter_10 = load_training_metrics("results_10_test/training_metrics/transform_tests/color_jitter.json")
no_transform_10 = load_training_metrics("results_10_test/training_metrics/transform_tests/no_transform.json")
random_gaussian_blue_10 = load_training_metrics("results_10_test/training_metrics/transform_tests/random_gaussian_blue.json")
random_grayscale_10 = load_training_metrics("results_10_test/training_metrics/transform_tests/random_grayscale.json")
random_rotation_10 = load_training_metrics("results_10_test/training_metrics/transform_tests/random_rotation.json")
random_vertical_flip_10 = load_training_metrics("results_10_test/training_metrics/transform_tests/random_vertical_flip.json")
random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10 = load_training_metrics("results_10_test/training_metrics/transform_tests/random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue.json")

plt.plot([x['epoch'] for x in color_jitter_10['training_metrics']], [x['validation_accuracy'] for x in color_jitter_10['training_metrics']], label="Color Jitter")
plt.plot([x['epoch'] for x in no_transform_10['training_metrics']], [x['validation_accuracy'] for x in no_transform_10['training_metrics']], label="No Transform")
plt.plot([x['epoch'] for x in random_gaussian_blue_10['training_metrics']], [x['validation_accuracy'] for x in random_gaussian_blue_10['training_metrics']], label="Random Gaussian Blue")
plt.plot([x['epoch'] for x in random_grayscale_10['training_metrics']], [x['validation_accuracy'] for x in random_grayscale_10['training_metrics']], label="Random Grayscale")
plt.plot([x['epoch'] for x in random_rotation_10['training_metrics']], [x['validation_accuracy'] for x in random_rotation_10['training_metrics']], label="Random Rotation")
plt.plot([x['epoch'] for x in random_vertical_flip_10['training_metrics']], [x['validation_accuracy'] for x in random_vertical_flip_10['training_metrics']], label="Random Vertical Flip")
plt.plot([x['epoch'] for x in random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10['training_metrics']], [x['validation_accuracy'] for x in random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10['training_metrics']], label="All Transforms")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Transform Tests")
plt.legend()
plt.savefig("./plots/transform_tests.png")

def smooth(training_metrics):
    training_metrics['training_metrics'][0]['smoothed_validation_accuracy'] = training_metrics['training_metrics'][0]['validation_accuracy']
    for i in range(1, len(training_metrics['training_metrics'])):
        training_metrics['training_metrics'][i]['smoothed_validation_accuracy'] = training_metrics['training_metrics'][i-1]['smoothed_validation_accuracy'] * 0.9 + training_metrics['training_metrics'][i]['validation_accuracy'] * 0.1
    return training_metrics

# function that does even lighter smoothing
def lighter_smooth(training_metrics):
    training_metrics['training_metrics'][0]['smoothed_validation_accuracy'] = training_metrics['training_metrics'][0]['validation_accuracy']
    for i in range(1, len(training_metrics['training_metrics'])):
        training_metrics['training_metrics'][i]['smoothed_validation_accuracy'] = training_metrics['training_metrics'][i-1]['smoothed_validation_accuracy'] * 0.99 + training_metrics['training_metrics'][i]['validation_accuracy'] * 0.01
    return training_metrics

# Same plot but with smoothed lines for all curves
color_jitter_10 = smooth(color_jitter_10)
no_transform_10 = smooth(no_transform_10)
random_gaussian_blue_10 = smooth(random_gaussian_blue_10)
random_grayscale_10 = smooth(random_grayscale_10)
random_rotation_10 = smooth(random_rotation_10)
random_vertical_flip_10 = smooth(random_vertical_flip_10)
random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10 = smooth(random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10)

plt.clf()
plt.plot([x['epoch'] for x in color_jitter_10['training_metrics']], [x['smoothed_validation_accuracy'] for x in color_jitter_10['training_metrics']], label="Color Jitter")
plt.plot([x['epoch'] for x in no_transform_10['training_metrics']], [x['smoothed_validation_accuracy'] for x in no_transform_10['training_metrics']], label="No Transform")
plt.plot([x['epoch'] for x in random_gaussian_blue_10['training_metrics']], [x['smoothed_validation_accuracy'] for x in random_gaussian_blue_10['training_metrics']], label="Random Gaussian Blue")
plt.plot([x['epoch'] for x in random_grayscale_10['training_metrics']], [x['smoothed_validation_accuracy'] for x in random_grayscale_10['training_metrics']], label="Random Grayscale")
plt.plot([x['epoch'] for x in random_rotation_10['training_metrics']], [x['smoothed_validation_accuracy'] for x in random_rotation_10['training_metrics']], label="Random Rotation")
plt.plot([x['epoch'] for x in random_vertical_flip_10['training_metrics']], [x['smoothed_validation_accuracy'] for x in random_vertical_flip_10['training_metrics']], label="Random Vertical Flip")
plt.plot([x['epoch'] for x in random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10['training_metrics']], [x['smoothed_validation_accuracy'] for x in random_vertical_flip_random_rotation_color_jitter_random_grayscale_random_gaussian_blue10['training_metrics']], label="Random Vertical Flip")


plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Transform Tests")
plt.legend()
plt.savefig("./plots/transform_tests_smoothed.png")

