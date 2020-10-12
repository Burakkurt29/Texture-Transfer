import numpy as np
import cv2
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument("-i1", "--texture_path", type=str)
parser.add_argument("-i2", "--target_path", type=str)
parser.add_argument("-b", "--block_size", type=int, default=50)
parser.add_argument("-o", "--overlap_size", type=int, default=10)
parser.add_argument("-m", "--color_mode", type=str, default='c')

args = parser.parse_args()

def minimum_cost_boundary(arr):
    arr_mask = np.ones(np.array(arr).shape)

    rows = len(arr)
    cols = len(arr[0])

    for i in range(1,rows):
        arr[i][0] = arr[i][0] + min(arr[i-1][0], arr[i-1][1])
        for j in range(1, cols-1):
            arr[i][j] = arr[i][j] + min(arr[i-1][j-1], arr[i-1][j], arr[i-1][j+1])
        arr[i][cols-1] = arr[i][cols-1] + min(arr[i-1][cols-2], arr[i-1][cols-1])


    min_index = [0]*rows
    min_cost = min(arr[-1])

    for k in range(1,cols-1):
        if arr[-1][k] == min_cost:
            min_index[-1] = k

    for i in range(rows-2, -1, -1):
        j = min_index[i+1]
        lower_bound = 0
        upper_bound = 1

        if j == cols-1:
            lower_bound = cols-2
            upper_bound = cols-1
        elif j > 0:
            lower_bound = j-1
            upper_bound = j+1

        min_cost = min(arr[i][lower_bound:upper_bound+1])

        for k in range(lower_bound, upper_bound+1):
            if arr[i][k] == min_cost:
                min_index[i] = k

    path = []
    for i in range(0, rows):
        arr_mask[i,0:min_index[i]] = np.zeros(min_index[i])
        path.append((i+1, min_index[i]+1))

    return arr_mask

def minimum_cost_mask(b, b1, b2, overlap_size, option):
    mask = np.ones(b.shape)
    if option == "vertical":
        arr = list(np.power(b1[:,-overlap_size:] - b[:,0:overlap_size], 2))
        mask[:,0:overlap_size] = minimum_cost_boundary(arr)
    elif option == "horizontal":
        arr = np.power(b2[-overlap_size:,:] - b[0:overlap_size,:], 2)
        arr = list(arr.transpose())
        mask[0:overlap_size, :] = minimum_cost_boundary(arr).transpose()
    elif option == "both":
        arr_vertical = list(np.power(b1[:,-overlap_size:] - b[:,0:overlap_size], 2))
        mask[:, 0:overlap_size] = minimum_cost_boundary(arr_vertical)
        arr_horizontal = np.power(b2[-overlap_size:,:] - b[0:overlap_size,:], 2)
        arr_horizontal = list(arr_horizontal.transpose())
        mask[0:overlap_size, :] = mask[0:overlap_size, :]*(minimum_cost_boundary(arr_horizontal).transpose())
    else:
        print("Error in min path")

    return mask

def transfer_color(texture, target, block_size, overlap_size):
    rows = texture.shape[0]
    cols = texture.shape[1]
    channels = texture.shape[2]

    out_size_row = target.shape[0]
    out_size_col = target.shape[1]

    blocks = []
    for i in range(rows - block_size):
        for j in range(cols - block_size):
            blocks.append(texture[i:i+block_size, j:j+block_size, :])


    result = np.ones([out_size_row, out_size_col, channels]) * -1
    result[0:block_size, 0:block_size, :] = texture[0:block_size, 0:block_size, :]

    num_blocks_row = 1 + np.ceil((out_size_row-block_size)*1.0 / (block_size-overlap_size))
    num_blocks_col = 1 + np.ceil((out_size_col-block_size)*1.0 / (block_size-overlap_size))

    for i in range(int(num_blocks_row)):
        for j in range(int(num_blocks_col)):
            if i == 0 and j == 0:
                continue

            start_row = int(i*(block_size - overlap_size))
            start_col = int(j*(block_size - overlap_size))
            end_row = int(min(start_row+block_size, out_size_row))
            end_col = int(min(start_col+block_size, out_size_col))

            fragment_result = result[start_row:end_row, start_col:end_col, :]
            target_block = target[start_row:end_row, start_col:end_col, :]

            match_block = match_color(blocks, fragment_result, target_block, block_size)

            b1_end_row = start_row + overlap_size - 1
            b1_start_row = b1_end_row - block_size + 1
            b1_end_col = start_col + overlap_size - 1
            b1_start_col = b1_end_col - block_size + 1

            if i == 0:
                b1 = result[start_row:end_row, b1_start_col:b1_end_col+1, :]
                mask = minimum_cost_mask(match_block[:, :, 0], b1[:, :, 0], 0, overlap_size, "vertical")
            elif j == 0:
                b2 = result[b1_start_row:b1_end_row+1, start_col:end_col]
                mask = minimum_cost_mask(match_block[:, :, 0], 0, b2[:, :, 0], overlap_size, "horizontal")
            else:
                b1 = result[start_row:end_row, b1_start_col:b1_end_col+1, :]
                b2 = result[b1_start_row:b1_end_row+1, start_col:end_col, :]
                mask = minimum_cost_mask(match_block[:, :, 0], b1[:, :, 0], b2[:, :, 0], overlap_size, "both")

            mask = np.repeat(np.expand_dims(mask, axis=2), 3, axis=2)
            mask_negate = mask == 0
            result[start_row:end_row, start_col:end_col, :] = mask_negate*result[start_row:end_row, start_col:end_col, :]
            result[start_row:end_row, start_col:end_col, :] = match_block*mask + result[start_row:end_row, start_col:end_col, :]

            completion = 100.0 / num_blocks_row*(i + j*1.0/num_blocks_col)
            print("{0:.2f}% complete".format(completion), end="\r", flush=True)

            if end_row == out_size_row:
                print("100% complete!", end="\r", flush = True)
                break

    return result

def transfer_gray(texture, target, block_size, overlap_size):
    rows = texture.shape[0]
    cols = texture.shape[1]

    out_size_row = target.shape[0]
    out_size_col = target.shape[1]

    blocks = []
    for i in range(rows - block_size):
        for j in range(cols - block_size):
            blocks.append(texture[i:i+block_size, j:j+block_size])

    result = np.ones([out_size_row, out_size_col]) * -1
    result[0:block_size, 0:block_size] = texture[0:block_size, 0:block_size]

    num_blocks_row = 1 + np.ceil((out_size_row-block_size)*1.0 / (block_size-overlap_size))
    num_blocks_col = 1 + np.ceil((out_size_col-block_size)*1.0 / (block_size-overlap_size))

    for i in range(int(num_blocks_row)):
        for j in range(int(num_blocks_col)):
            if i == 0 and j == 0:
                continue

            start_row = int(i*(block_size - overlap_size))
            start_col = int(j*(block_size - overlap_size))
            end_row = int(min(start_row+block_size, out_size_row))
            end_col = int(min(start_col+block_size, out_size_col))
            fragment_result = result[start_row:end_row, start_col:end_col]
            target_block = target[start_row:end_row, start_col:end_col]
            match_block = match_gray(blocks, fragment_result, target_block, block_size)

            b1_end_row = start_row + overlap_size - 1
            b1_start_row = b1_end_row - block_size + 1
            b1_end_col = start_col + overlap_size - 1
            b1_start_col = b1_end_col - block_size + 1

            if i == 0:
                b1 = result[start_row:end_row, b1_start_col:b1_end_col+1]
                mask = minimum_cost_mask(match_block[:], b1[:], 0, overlap_size, "vertical")
            elif j == 0:
                b2 = result[b1_start_row:b1_end_row+1, start_col:end_col]
                mask = minimum_cost_mask(match_block[:], 0, b2[:], overlap_size, "horizontal")
            else:
                b1 = result[start_row:end_row, b1_start_col:b1_end_col+1]
                b2 = result[b1_start_row:b1_end_row+1, start_col:end_col]
                mask = minimum_cost_mask(match_block[:], b1[:], b2[:], overlap_size, "both")

            mask_negate = mask == 0
            result[start_row:end_row, start_col:end_col] = mask_negate*result[start_row:end_row, start_col:end_col]
            result[start_row:end_row, start_col:end_col] = match_block*mask + result[start_row:end_row, start_col:end_col]

            completion = 100.0 / num_blocks_row*(i + j*1.0/num_blocks_col)
            print("{0:.2f}% complete".format(completion), end="\r", flush=True)

            if end_row == out_size_row:
                print("100% complete!", end="\r", flush = True)
                break

    return result

def match_color(blocks, fragment, target_block, block_size):
    tolerance = 0.1
    rows = fragment.shape[0]
    cols = fragment.shape[1]
    channels = fragment.shape[2]
    errors = []
    best_blocks = []

    for i in range(len(blocks)):
        bi = blocks[i][0:rows, 0:cols, 0:channels]
        errors.append(ssd(bi, fragment, target_block, 3))

    min_val = np.min(errors)
    for i, block in enumerate(blocks):
        if errors[i] <= (1.0+tolerance)*min_val:
            best_blocks.append(block[:rows, :cols, :channels])

    rand_int = np.random.randint(len(best_blocks))
    return best_blocks[rand_int]

def match_gray(blocks, fragment, target_block, block_size):
    tolerance = 0.1
    rows = fragment.shape[0]
    cols = fragment.shape[1]
    errors = []
    best_blocks = []

    for i in range(len(blocks)):
        bi = blocks[i][0:rows,0:cols]
        errors.append(ssd(bi, fragment, target_block, 2))

    min_val = np.min(errors)
    for i, block in enumerate(blocks):
        if errors[i] <= (1.0+tolerance)*min_val:
            best_blocks.append(block[:rows, :cols])

    rand_int = np.random.randint(len(best_blocks))
    return best_blocks[rand_int]

def ssd(block, fragment, target, dim):
    alpha = 0.1

    if dim == 2:
        lum_block = np.sum(block, axis=1)*1.0/3
        lum_target = np.sum(target, axis=1)*1.0/3
        lum_fragment = np.sum(fragment, axis=1)*1.0/3
    elif dim == 3:
        lum_block = np.sum(block, axis=2)*1.0/3
        lum_target = np.sum(target, axis=2)*1.0/3
        lum_fragment = np.sum(fragment, axis=2)*1.0/3

    error = alpha*np.sqrt(np.sum(((fragment+0.99)>0.1)*(block-fragment)*(block-fragment))) + (1-alpha)*np.sqrt(np.sum(((lum_fragment+0.99)>0.1)*(lum_block-lum_target)*(lum_block-lum_target)))

    return error

if __name__ == "__main__":
    try:
        texture = cv2.imread(args.texture_path, cv2.IMREAD_COLOR)
        target = cv2.imread(args.target_path, cv2.IMREAD_COLOR)
        if args.color_mode == 'c':
            result = transfer_color(texture, target, args.block_size, args.overlap_size)
            result_path = "./results/" + args.texture_path.split("/")[-1].split(".")[0] + "_to_" + args.target_path.split("/")[-1].split(".")[0]+"_color_"
        elif args.color_mode == 'g':
            texture = cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY)
            target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            result = transfer_gray(texture, target, args.block_size, args.overlap_size)
            result_path = "./results/" + args.texture_path.split("/")[-1].split(".")[0] + "_to_" + args.target_path.split("/")[-1].split(".")[0]+"_gray_"

        result_path += "b=" + str(args.block_size) + "_o=" + str(args.overlap_size) + ".jpg"
        cv2.imwrite(result_path, result)

    except Exception as e:
        print("Error : ", e)
        sys.exit(1)
