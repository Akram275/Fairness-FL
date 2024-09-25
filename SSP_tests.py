import numpy as np
import time


def approximate_subset_sum_floats(nums, target, epsilon, precision=4):
    """
    Approximate solution to the Subset Sum problem with float values.

    :param nums: List of float numbers.
    :param target: Target sum (float).
    :param epsilon: Approximation factor (float).
    :param precision: Number of decimal places for precision.
    :return: Approximate subset that sums to a value close to target.
    """
    scale = 10 ** precision
    scaled_nums = [int(num * scale) for num in nums]
    scaled_target = int(target * scale)
    scaled_epsilon = int(epsilon * scale)

    def trim(sums, delta):
        sums = sorted(sums)
        trimmed = [sums[0]]
        last = sums[0]
        for s in sums[1:]:
            if s > last * (1 + delta):
                trimmed.append(s)
                last = s
        return trimmed

    # Initialize achievable sums and their corresponding subsets
    achievable_sums = {0: []}

    for i, num in enumerate(scaled_nums):
        new_sums = {}
        for s in achievable_sums:
            new_sum = s + num
            if new_sum <= scaled_target + scaled_epsilon:
                new_subset = achievable_sums[s] + [i]
                if new_sum in new_sums:
                    if len(new_subset) < len(new_sums[new_sum]):
                        new_sums[new_sum] = new_subset
                else:
                    new_sums[new_sum] = new_subset
        achievable_sums.update(new_sums)
        trimmed_sums = trim(list(achievable_sums.keys()), epsilon / (2 * len(nums)))
        achievable_sums = {s: achievable_sums[s] for s in trimmed_sums}

    closest_sum = max((s for s in achievable_sums if scaled_target - scaled_epsilon <= s <= scaled_target + scaled_epsilon), default=None)

    if closest_sum is not None:
        subset = achievable_sums[closest_sum]
        closest_sum /= scale
        return closest_sum, subset

    return None, []


if __name__=="__main__" :

    # Example usage
    nums = np.random.normal(0, 0.1, 100).tolist()
    target = 0
    epsilon = 0.0001

    start_time = time.time()  # Record the start time
    result_sum, result_subset = approximate_subset_sum_floats(nums, target, epsilon)
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time

    print(f"Approximate sum within [{target - epsilon}, {target + epsilon}]: {result_sum}")
    print(f"Subset that achieves the approximate sum: {result_subset}")
    print(f"Execution time: {execution_time * 1000} miliseconds")
