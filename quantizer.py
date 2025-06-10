from itertools import count

import torch
import numpy as np
from matplotlib import pyplot as plt
import time



class LatticeQuantization:
    def __init__(self, args, hex_mat, our_round=True):
        self.gamma = args.gamma
        self.args = args
        self.our_round = our_round
        self.round = self.sigm if self.our_round else torch.round
        self.dim = args.lattice_dim
        self.gen_mat = hex_mat
        self.num_codewords = args.num_codewords
        self.overloading = args.num_overloading

        # Estimate P0_cov
        self.delta = (2 * args.gamma) / (2 ** args.R + 1)
        orthog_domain_dither = torch.from_numpy(
            np.random.uniform(low=-self.delta / 2, high=self.delta / 2, size=[args.lattice_dim, 1000])
        ).float()
        device = self.gen_mat.device
        orthog_domain_dither = orthog_domain_dither.to(device)
        lattice_domain_dither = torch.matmul(self.gen_mat, orthog_domain_dither)
        self.P0_cov = torch.cov(lattice_domain_dither)

    def sigm(self, x):
        """Differentiable rounding operation."""
        return x + 0.2 * (torch.cos(2 * torch.pi * (x + 0.25)))

    def divide_into_blocks(self, input, dim=2):
        """Divide input into blocks of a given dimension, zero-padded if needed."""
        input = input.view(-1)
        pad_with = (dim - len(input) % dim) % dim
        if pad_with > 0:
            input = torch.cat((input, torch.zeros(pad_with, dtype=input.dtype, device=input.device)))
        return input.view(dim, -1), pad_with

    def combine_blocks(self, blocks, pad_with, original_shape):
        """Combine blocks back into the original tensor shape, removing padding."""
        flat = blocks.reshape(-1) # Flatten the blocks
        if pad_with > 0:
            flat = flat[:-pad_with]  # Remove padding
        return flat.view(original_shape)  # Reshape back to original shape

    def print_input_after_blocks(self, input):
        # Get the shape of the tensor
        # Extract non-zero indices and values
        print(input)
        indices = torch.nonzero(input)
        values = input[indices[:, 0], indices[:, 1]]
        x_coords = []
        y_coords = []
        shape = input.shape
        M = shape[1]
        # Extract x and y coordinates
        for i in range(M):
            x_coords.append(input[0, i])
            y_coords.append(input[1, i])
        print(x_coords)
        print(y_coords)

        # Plot the points
        plt.scatter(x_coords, y_coords)
        plt.colorbar()
        plt.show()
        time.sleep(2)

    def convert_vec_to_show(self, vec):
        indices = torch.nonzero(vec)
        x_coords = []
        y_coords = []
        shape = vec.shape
        M = shape[1]
        # Extract x and y coordinates
        for i in range(M):
            x_coords.append(vec[0, i])
            y_coords.append(vec[1, i])
        return  x_coords, y_coords

    def scale_points_to_fit_circle(self, points, for_grid=True, target_count=23, should_print =False, radius=1):
        """
        Scale points so that only `target_count` points fit within a circle of the given `radius`.
        :param points: Array of points (N x 2).
        :param target_count: Number of points to fit inside the circle.
        :param radius: Radius of the circle.
        :return: Scaled points, scaling factor.
        """
        # Compute distances from the origin
        shouldAddVar = True
        if for_grid == False :
            points = points.T
            target_count = 1000000000000000
        if for_grid:
            # Create a mask for unique points
            is_unique = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)

            for i in range(points.shape[0]):
                if is_unique[i]:
                    # Compare exact point values, mark duplicates as False
                    is_unique[(points == points[i]).all(dim=1) & (is_unique)] = False
                    # Ensure the current point remains True
                    is_unique[i] = True

            # Return unique points
            points = points[is_unique]
        distances = torch.linalg.norm(points, dim=1)
        if for_grid == False and shouldAddVar:


            # Sort distances and get indices
            sorted_distances, indices = torch.sort(distances, descending=True)

            # Keep removing the largest distances if variance is too high
            remaining_distances = sorted_distances.clone()
            counter = 0
            # # Define the variance threshold
            if self.overloading == -1:
                variance_threshold = 0.003  # Adjust based on your use case
                max_counter = 10
                while len(remaining_distances) > 1 and torch.var(remaining_distances) > variance_threshold and counter <= 10:
                    remaining_distances =remaining_distances[1:]  # Remove the largest distance
                    counter = counter +1
            else:
                precent = self.overloading
                max_counter =precent*len(sorted_distances)/100
                while len(remaining_distances) > 1 and counter <= max_counter:
                    remaining_distances =remaining_distances[1:]  # Remove the largest distance
                    counter = counter +1
            print(f"num of points I removed {counter}")
        else:
            remaining_distances = distances

        threshold_distance = remaining_distances.topk(target_count, largest=False).values[-1] if target_count < len(remaining_distances) else remaining_distances.max()

        if should_print:
            print(f"distance {threshold_distance}")

        # Scale the points
        scaled_points = points / threshold_distance

        return scaled_points, threshold_distance

    def plot_codeword_graph(self, codeee, blue_dots, connections, radius=1):
        # Create a figure
        fig, ax = plt.subplots(figsize=(8, 8))

        # Plot the circle
        circle = plt.Circle((0, 0), radius, color='red', fill=False, linewidth=2)
        ax.add_artist(circle)

        # Plot red dots (codewords)
        codewords = np.array(codeee)
        plt.scatter(codewords[:, 0], codewords[:, 1], color='red', label='Codewords')

        # Plot blue dots
        blue_dots = np.array(blue_dots)
        plt.scatter(blue_dots[:, 0], blue_dots[:, 1], color='blue', label='Blue Dots')

        # Add connections (green lines) between blue dots and red dots
        for blue, red in zip(blue_dots, connections):
            plt.plot([blue[0], red[0]], [blue[1], red[1]], color='green', linewidth=1)

        # Plot evenly spaced black lines between codewords
        min_x, max_x = -5, 5
        min_y, max_y = -5, 5
        for i in range(len(codewords)):
            for j in range(i + 1, len(codewords)):
                midpoint = (codewords[i] + codewords[j]) / 2
                direction = np.array([codewords[j][1] - codewords[i][1], -(codewords[j][0] - codewords[i][0])],
                                     dtype=float)
                if np.linalg.norm(direction) != 0:
                    direction /= np.linalg.norm(direction)
                start = midpoint - 10 * direction
                end = midpoint + 10 * direction
                # plt.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=0.5)

        # Set axis limits and labels
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(False)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend()
        plt.show()

    def __call__(self, input, shouldPrint = False, shouldReturnBack = False, gettingAlph = None):
        #shouldPrint = False
        c1 = (0.0, 0.0)  # Circle center (x, y)
        shouldHaveDither = False
        circle_radius = 1  # Circle radius
        G = self.gen_mat
        original_shape = input.shape
        vec, pad_with = self.divide_into_blocks(input, self.args.lattice_dim) #todo convert this back in FL
        
        if shouldHaveDither:
            dither = torch.zeros_like(vec, dtype=torch.float32)
            # print(self.delta)
            dither = dither.uniform_(-self.delta / 2, self.delta / 2)  # generate dither
        else:
            dither = 0
        if gettingAlph is not None:
            scaling_factor_vec = gettingAlph
            scaled_points_vec = (vec + dither)/scaling_factor_vec
        else: 
            scaled_points_vec, scaling_factor_vec = self.scale_points_to_fit_circle(vec + dither, False, 1000000000000000 , shouldPrint)
            scaled_points_vec = scaled_points_vec.T
        if shouldPrint:
            print(f"vec before anything: {input}")
            print(f"vec after divide into blocks and remove 0 0 : {vec}")
            print(f"vec  dither: {dither}")
            print(f"vec after normilizing {scaled_points_vec}")

        ranges = [torch.arange(-100, 101) for _ in range(self.args.lattice_dim)]
        grid = torch.stack(torch.meshgrid(*ranges, indexing="ij"), dim=-1).reshape(-1, self.args.lattice_dim)
        device = self.gen_mat.device
        grid = grid.to(device)
        transformed_points = torch.matmul(grid.float(), self.gen_mat.T)
        codewords, scaling_factor = self.scale_points_to_fit_circle(transformed_points, True, self.num_codewords)  
        if shouldPrint:
            print(f"grid : {grid}")
            print(f"G before c: {G}")

        # Filter codewords to fit within the circle
        distances = torch.linalg.norm(codewords, dim=1)
        filtered_codewords = codewords[distances <= 1]

        if shouldPrint:
            print(f"G after c: {G}, c is: {scaling_factor}")
            print(f"codewords before filter {codewords}")
            print(f"codewords after filter {filtered_codewords}, len {len(filtered_codewords)}")
        codewords = filtered_codewords

        # Assignments of input

        device = scaled_points_vec.device
        codewords = codewords.to(device)
        print(len(codewords))

        distances = torch.cdist(scaled_points_vec.T.to(torch.float32), codewords.to(torch.float32))
        if distances.size(1) == 0:
            print(distances)
            print(codewords)
            print(self.gen_mat)
            print(scaling_factor)
            return codewords, vec
        assignments = distances.argmin(dim=1)
        reconstructed_points = codewords[assignments].T

        if shouldPrint:
            self.plot_codeword_graph(codewords, scaled_points_vec.T, reconstructed_points.T)

        vec1 = scaled_points_vec
        output =((reconstructed_points * scaling_factor_vec)- dither).to(torch.float32)
        if shouldPrint:
            print(f"output: {output}")
        if shouldPrint:
            x_coords, y_coords = self.convert_vec_to_show(vec)
            x_coords_vec_changed, y_coords_vec_changed = self.convert_vec_to_show(scaled_points_vec)
            x_coords_vec_after_q, y_coords_vec_after_q = self.convert_vec_to_show(torch.tensor(codewords, dtype=torch.float32))
            x_coords_vec_after_q_norm, y_coords_vec_after_q_norm = self.convert_vec_to_show(output)
            x_coords_code, y_coords_code = self.convert_vec_to_show(torch.tensor( codewords, dtype=torch.float32).T)
            # # Plot the points
            #plt.scatter(x_coords, y_coords,color='black', label='vec')
            plt.scatter(x_coords_vec_changed, y_coords_vec_changed, color='green', label='input_after_norm')
            #plt.scatter(x_coords_vec_after_q, y_coords_vec_after_q, color='red', label='after_quantize')
            plt.scatter(x_coords_vec_after_q_norm, y_coords_vec_after_q_norm, color='gray', label='after_quantize_and_norm_back')
            plt.scatter(x_coords_code, y_coords_code, color='blue', label='codeword')
            circle = plt.Circle(c1, circle_radius, color='yellow', fill=False, label='circle c1')
            plt.gca().add_patch(circle)
            plt.legend()
            plt.show()
            time.sleep(2)
            ######################
            signal_power = torch.mean((vec) ** 2)  # Power of the signal
            noise_power = torch.mean((vec- output) ** 2)  # Power of the noise
            snr = 10 * torch.log10(signal_power / noise_power)  # SNR in decibels
            print(f"snr : {snr}")
        if shouldReturnBack:
            reconstructed_tensor = self.combine_blocks(output, pad_with, original_shape)
            return reconstructed_tensor, input
        return output,vec.to(torch.float32)


