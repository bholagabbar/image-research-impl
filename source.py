import cv2
import math
import binascii
import argparse
import numpy as np
from matplotlib import pyplot as plt

'''UTILITY FUNCTIONS
'''


def odd(val):
    return bool(val % 2)


def even(val):
    return not bool(val % 2)


def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))


def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)


def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))


# return Root Mean Square values between 2 blocks
def rmse(predictions, targets):
    return np.sqrt(np.mean((np.array(predictions) - np.array(targets)) ** 2))


'''AUXILIARY METHODS
'''


# breaks into blocks with {indices, block, standard deviation, mean}
def break_ip_img_into_blocks_with_data(image):
    width, height = image.shape
    blocks = dict()
    block_cnt = 0
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            curr_block = image[y:y + 4, x:x + 4]
            # Storing 1. count 2. image block 3. SD for block 4. Mean for block
            blocks[block_cnt] = (block_cnt, curr_block, np.std(curr_block), np.mean(curr_block))
            # print blocks[block_cnt][2]
            block_cnt += 1
    return blocks


# classify as 0,1 and make parings
def get_pair_mapping(ip_block_mapping, tg_block_mapping):
    # get blocks
    ip_blocks = ip_block_mapping.values()
    tg_blocks = tg_block_mapping.values()
    # sort wrt to SD
    ip_blocks.sort(key=lambda x: x[2])
    tg_blocks.sort(key=lambda x: x[2])
    # pair wrt CIT
    alpha = 0.7
    n_alpha = int(alpha * len(ip_blocks))
    pairings = dict()
    pairings[0] = list()
    pairings[1] = list()
    for i in xrange(n_alpha):
        pairings[0].append((ip_blocks[i][0], tg_blocks[i][0]))
    for i in xrange(n_alpha, len(ip_blocks)):
        pairings[1].append((ip_blocks[i][0], tg_blocks[i][0]))
    return pairings


# adjust over/underflow in blocks while adding delta-u
def adjust_u_with_ov_un(u, block):
    ov = -1
    un = 256
    for i in xrange(0, 4):
        for j in xrange(0, 4):
            if block[i][j] > ov:
                ov = block[i][j]
            if block[i][j] < un:
                un = block[i][j]
    if u >= 0 and ov + u > 255:
        ov -= u
        u = u + 255 - ov
    elif u < 0 and un + u < 0:
        un += u
        u -= un
    return u


# PROBLEMATIC. Further adjust delta-u with lamda to aid compression of range
def adjust_u_with_lamda(u):
    # -ve u does not work
    if u < 0:
        return u
    lamda = 2
    if u >= 0:
        u = lamda * round(u / lamda)
    elif u < 0:
        u = lamda * math.floor(u / lamda) + (lamda / 2)
    u_dash = (2 * math.fabs(u)) / lamda
    return u_dash


def get_delta_u(ip_block, tg_block):
    # difference of means of target and ip block
    u = round(tg_block[3] - ip_block[3])
    # adjust overflow/underflow and further with lamda to compress
    u = adjust_u_with_ov_un(u, ip_block[1])
    u = adjust_u_with_lamda(u)
    return int(u)


# first create target block, rotate nblock to minimize MSE between rotated variation of this and target
def transform_to_t_dash(ip_block_mapping, tg_block_mapping, pair_mapping):
    tf_blocks = dict()
    tf_block_data = dict()
    # 0 and 1
    for pair_no in xrange(0, 2):
        for mapping in pair_mapping[pair_no]:
            # Get blocks of respective mapping from mapping pair
            ip_block = ip_block_mapping[mapping[0]]
            tg_block = tg_block_mapping[mapping[1]]
            new_block = [[], [], [], []]
            # get delta_u for the block
            delta_u = get_delta_u(ip_block, tg_block)
            for j in xrange(0, 4):
                for k in xrange(0, 4):
                    val = ip_block[1][j][k] + delta_u
                    new_block[j].append(val)
            # check rotation and RMSE
            rotated = np.array(new_block)
            final = rotated
            min_rmse = rmse(rotated, tg_block[1])
            degree = 0
            # 90, 180, 270 rotations, store minimum
            for i in xrange(1, 4):
                rotated = np.rot90(rotated, 1)
                curr_rmse = rmse(rotated, tg_block[1])
                if curr_rmse < min_rmse:
                    min_rmse = curr_rmse
                    final = rotated
                    degree = i
            # since Ti with T'i (CIT/table for Transformed and target is same)
            tf_blocks[tg_block[0]] = final
            # Store delta_u, degree of rotation
            tf_block_data[tg_block[0]] = (delta_u, degree)
    return tf_blocks, tf_block_data


def break_tg_img_into_blocks(image):
    width, height = image.shape
    blocks = dict()
    block_cnt = 0
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            curr_block = image[y:y + 4, x:x + 4]
            blocks[block_cnt] = curr_block
            block_cnt += 1
    return blocks


def build_image_from_blocks(img_blocks):
    n_img = np.zeros((512, 512), dtype='uint8')
    block_no = 0
    for i in xrange(0, 512, 4):
        for j in xrange(0, 512, 4):
            n_img[i:i + 4, j:j + 4] = img_blocks[block_no]
            block_no += 1
    return n_img


def display_image(img):
    cv2.imshow('Input Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


'''PRIMARY METHODS
'''


# PHASE 1
def transform_input_image(ip_img, tg_img):
    # resize
    # ip_img = cv2.resize(ip_img, (512, 512), interpolation=cv2.INTER_CUBIC)
    # tg_img = cv2.resize(tg_img, (512, 512), interpolation=cv2.INTER_CUBIC)

    # get blocks of input and target image, sorted wrt to std
    ip_block_mapping = break_ip_img_into_blocks_with_data(ip_img)
    tg_block_mapping = break_ip_img_into_blocks_with_data(tg_img)

    # get pairings as 0, 1 paired
    pair_mapping = get_pair_mapping(ip_block_mapping, tg_block_mapping)

    # get transformed blocks
    tf_blocks, tf_block_data = transform_to_t_dash(ip_block_mapping, tg_block_mapping, pair_mapping)

    tf_img = build_image_from_blocks(tf_blocks)
    return tf_img, pair_mapping, tf_block_data


# PHASE 2
def embed_msg_and_mod_transformed_img(img, msg):
    enc_img = img.copy()
    msg = text_to_bits(msg)
    pair_changes = list()
    cnt = 0
    for i in range(len(img)):
        for j in range(len(img[i])):
            if cnt < len(msg):
                # See encoding_logic.txt file for logic
                curr_bit = int(msg[cnt])
                curr_pixel = img[i][j]
                # j is col index, curr_pixel and curr_bit as is
                if ((curr_bit == 1 and odd(j) and odd(curr_pixel)) or
                        (curr_bit == 1 and even(j) and even(curr_pixel)) or
                        (curr_bit == 0 and odd(j) and even(curr_pixel)) or
                        (curr_bit == 0 and even(j) and odd(curr_pixel))):
                    # No change in image bit
                    pair_changes.append((i, j, False))
                else:
                    pair_changes.append((i, j, True))
                    # Add 1 to pixel
                    enc_img[i][j] += 1
                cnt += 1
            else:
                return enc_img, pair_changes


# PHASE 3
def extract_msg_and_restore_to_transformed_img(img, pair_changes):
    dec_bin_msg = ''
    for values in pair_changes:
        row, col, has_changed = values[0], values[1], values[2]
        # Restore bit is it has been changed (essentially, +1)
        if has_changed is True:
            img[row][col] -= 1
        # Restored Pixel
        res_pix = img[row][col]
        # Check condition and restore message
        if ((odd(col) and odd(res_pix) and not has_changed) or
                (even(col) and even(res_pix) and not has_changed) or
                (odd(col) and even(res_pix) and has_changed) or
                (even(col) and odd(res_pix) and has_changed)):
            dec_bin_msg += '1'
        else:
            dec_bin_msg += '0'
    dec_txt_msg = text_from_bits(dec_bin_msg)
    return img, dec_txt_msg


# PHASE 4
def restore_to_input_img(tf_img, pair_mapping, tf_block_data):
    tf_blocks = break_tg_img_into_blocks(tf_img)
    ip_blocks = [None] * len(tf_blocks)
    ip_to_tg_mapping = dict()

    for index in xrange(len(pair_mapping)):
        for mapping in pair_mapping[index]:
            ip_to_tg_mapping[mapping[0]] = mapping[1]

    # Mod transformed blocks back to ip
    for i in xrange(len(tf_blocks)):
        # Mapping index of input block to target block
        curr_mapping_index = ip_to_tg_mapping[i]
        # Get the corresponding target block
        curr_tg_block = tf_blocks[curr_mapping_index]
        delta_u, degree = tf_block_data[curr_mapping_index]
        # Subtract delta_u from every pixel
        for j in xrange(0, 4):
            for k in xrange(0, 4):
                curr_tg_block[j][k] -= delta_u
        # Rotate by 360 minus degree of rotation of tf_block
        curr_tg_block = np.rot90(curr_tg_block, 4 - degree)
        ip_blocks[i] = curr_tg_block
    f_ori_img = build_image_from_blocks(ip_blocks)
    return f_ori_img


'''DRIVER
'''


# Execute Flow of Phases
def main(ip_img, tg_img, msg):
    # Transform Image
    tf_img, pair_mapping, tf_block_data = transform_input_image(ip_img, tg_img)
    # Embed Message in Image, get Location Matrix values while modding Transformed Image
    enc_tf_img, pair_changes = embed_msg_and_mod_transformed_img(tf_img, msg)
    # Retrieve Message from Image, restore to Original Transformed Image
    dec_tf_img, dec_txt_msg = extract_msg_and_restore_to_transformed_img(enc_tf_img.copy(), pair_changes)
    # Restore Original image
    res_ip_img = restore_to_input_img(dec_tf_img.copy(), pair_mapping, tf_block_data)
    # Plot resultant images
    img_list = [ip_img, tg_img, tf_img, enc_tf_img, dec_tf_img, res_ip_img]
    img_title_list = ['Input', 'Target', 'Transformed', 'Encoded Transformed',
                      'Decoded Transformed', 'Restored Input']
    for i in xrange(len(img_list)):
        plt.subplot(2, 3, i + 1), plt.imshow(img_list[i], 'gray')
        plt.title(img_title_list[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


# Take Input
parser = argparse.ArgumentParser()
parser.add_argument('--input', nargs='?', default='resources/images/lena_gray_512.tif')
parser.add_argument('--target', nargs='?', default='resources/images/woman_darkhair.tif')
parser.add_argument('--msg', nargs='?', default='hello')
args = parser.parse_args()
# Load Images
input_image = cv2.imread(args.input, cv2.CV_LOAD_IMAGE_GRAYSCALE)
target_image = cv2.imread(args.target, cv2.CV_LOAD_IMAGE_GRAYSCALE)
# Run Driver
main(input_image, target_image, args.msg)
