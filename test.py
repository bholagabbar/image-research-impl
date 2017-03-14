import numpy as np
import random
import cv2
import os
import source


def generate_percent_of_payload(payload_size):
    rand_char_str = str()
    for i in xrange(payload_size):
        rand_char_str += chr(random.randint(97, 122))
    return rand_char_str


# Asserts whether final constructed input image and initial images are same, Runs payload tests for different capacities
def run_payloads_for_images_and_assert_final_image_equal_to_input(ip_img_path, tg_img_path, which_images):
    # No of characters in 10%, 20%, 40%, 50%, 60%, 80% of 512x512. Calculated using generate_percent_of_payload function
    payloads = [3277, 6554, 13108, 16384, 19661, 26215]
    payload_amount = ['payload_10', 'payload_20', 'payload_40', 'payload_50', 'payload_60', 'payload_80']
    # Create Directory
    path_prefix = 'test_results/' + str(which_images)
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    ip_img = cv2.imread(ip_img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    tg_img = cv2.imread(tg_img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imwrite(path_prefix + '/input.pgm', ip_img)
    cv2.imwrite(path_prefix + '/target.pgm', tg_img)
    # Different payloads testing
    for i in xrange(len(payloads)):
        msg = generate_percent_of_payload(payloads[i])
        # Execute same flow as main
        tf_img, pair_mapping, tf_block_data = source.transform_input_image(ip_img.copy(), tg_img.copy())
        enc_tf_img, pair_changes = source.embed_msg_and_mod_transformed_img(tf_img, msg)
        dec_tf_img, dec_txt_msg = source.extract_msg_and_restore_to_transformed_img(enc_tf_img.copy(), pair_changes)
        res_ip_img = source.restore_to_input_img(dec_tf_img.copy(), pair_mapping, tf_block_data)
        # Assert equality
        assert np.array_equal(ip_img, res_ip_img)
        # Append payload
        path_prefix = 'test_results/' + str(which_images) + '/' + payload_amount[i]
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        # For comparision
        cv2.imwrite(path_prefix + '/target_img.pgm', tg_img)
        cv2.imwrite(path_prefix + '/stego_img.pgm', tf_img)
        cv2.imwrite(path_prefix + '/encoded_img.pgm', enc_tf_img)


# All images are numbered as consecutive natural nos.
def pair_all_images(no_of_images):
    img_path_prefix = 'resources/images/'
    cnt = 0
    for i in xrange(1, no_of_images + 1):
        ip_img_path = img_path_prefix + str(i) + '.pgm'
        for j in xrange(i + 1, no_of_images + 1):
            tg_img_path = img_path_prefix + str(j) + '.pgm'
            try:
                which_images = str(i) + '_' + str(j)
                run_payloads_for_images_and_assert_final_image_equal_to_input(ip_img_path, tg_img_path, which_images)
                print which_images, 'Completed'
            except Exception as e:
                print e
            cnt += 1
    print cnt


# Run brute pairing of images
pair_all_images(100)
