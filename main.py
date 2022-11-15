import cv2 as cv
import matplotlib.pyplot as plt

from image import Image
from k_means import KMeans
from particle_swarm import ParticleSwarm


def main(image_name: str) -> None:
    path = f"./assets/{image_name}.jpg"
    image = Image(path=path)

    image = image.resize(height=408, width=408)
    image_shape = image.shape

    image = image.reshape((-1, 3))
    image = image.to_float_32()

    upper_bound = [5, 5, 5, 1]
    lower_bound = [1, 1, 1, 0.1]

    k_means = KMeans(image, image_shape)

    pso = ParticleSwarm(k_means.run, upper_bound, lower_bound)
    pso.run()

    show_results(image_name, pso.result, k_means)


def show_results(image_name: str, pso_result: str, k_means: KMeans):
    print(pso_result)

    history = k_means.history
    attempts = list(range(len(history)))

    plt.title(f"{image_name.capitalize()} Convergence")
    plt.plot(attempts, history, "--")
    plt.yscale("log")
    plt.show()

    cv.imshow(f"Clustered {image_name.capitalize()}", k_means.clustered_image)
    cv.waitKey(0)


if __name__ == '__main__':
    images = ["nature", "safari", "scarface"]

    for name in images:
        main(name)
