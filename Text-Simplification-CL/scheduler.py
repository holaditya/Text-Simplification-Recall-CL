import random
import logging


def get_weights(i, n):
    weights_by_step = {
        0: [0.7, 0.2, 0.1],
        n // 3: [0.1, 0.7, 0.2],
        2 * n // 3: [0.2, 0.1, 0.7],
    }
    # weights_by_step = {
    #     0: [1, 0, 0],
    #     n // 3: [0, 1, 0],
    #     2 * n // 3: [0, 0, 1],
    # }

    if i < 0:
        raise ValueError("i must be non-negative")
    if i > 0 and i < n // 3:
        return weights_by_step[0]
    elif i >= n // 3 and i < 2 * n // 3:
        return weights_by_step[n // 3]
    else:
        return weights_by_step[2 * n // 3]


def choose_number(n=100):
    # Let's initialize initial weights based on the number of batches in each loader
    weights_by_step = {
        0: [0.7, 0.2, 0.1],
        n // 3: [0.1, 0.7, 0.2],
        2 * n // 3: [0.2, 0.1, 0.7],
    }

    choices = []
    for i in range(n):
        # Choose a number based on weights
        weights = get_weights(i, n)
        chosen_number = random.choices([1, 2, 3], weights)[0]
        choices.append(chosen_number)
        yield chosen_number

        # Normalize weights to ensure they sum to 1

    logging.info(f"Choices: {choices}")
    return choices


def get_batches(l1_loader, l2_loader, l3_loader):
    loaders = {
        1: iter(l1_loader),
        2: iter(l2_loader),
        3: iter(l3_loader)
    }

    finished = {
        1: False,
        2: False,
        3: False
    }

    n = len(l1_loader) + len(l2_loader) + len(l3_loader)
    choices = []
    for choice in choose_number(n):
        returned = False
        while not returned:
            try:
                p = next(loaders[choice])
                returned = True
                yield p
            except StopIteration:
                finished[choice] = True
                # Chose a random number from the remaining iterators
                choice = random.choice(list(loaders.keys()))
                # If all iterators are exhausted, stop
            finally:
                choices.append(choice)

        if all(finished.values()):
            break
    logging.info(f"Actual Choices: {choices}")
