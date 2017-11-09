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


def quick_generate_stego_targets_for_comparision(ip_img_path, tg_img_path, which_images):
    path_prefix = 'test_results/'
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    ip_img, tg_img, tf_img, enc_tf_img, dec_tf_img, res_ip_img = source.main(ip_img_path, tg_img_path, 'hello',
                                                                             do_return=True, show_plot=False)
    cv2.imwrite(path_prefix + str(which_images) + '_target.pgm', tg_img)
    cv2.imwrite(path_prefix + str(which_images) + '_stego.pgm', tf_img)


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
                # run_payloads_for_images_and_assert_final_image_equal_to_input(ip_img_path, tg_img_path, which_images)
                quick_generate_stego_targets_for_comparision(ip_img_path, tg_img_path, which_images)
                print which_images, 'Completed'
            except Exception as e:
                print e
            cnt += 1
    print cnt


def pair_images_passed(image_list, left, right):
    img_path_prefix = 'resources/images/'
    cnt = 0
    for i in image_list:
        ip_img_path = img_path_prefix + str(i) + '.pgm'
        for j in range(left, right + 1):
            tg_img_path = img_path_prefix + str(j) + '.pgm'
            try:
                which_images = str(i) + '_' + str(j)
                quick_generate_stego_targets_for_comparision(ip_img_path, tg_img_path, which_images)
                print which_images, 'Completed'
                if j not in image_list and j is not i:
                    which_images = str(j) + '_' + str(i)
                    quick_generate_stego_targets_for_comparision(tg_img_path, ip_img_path, which_images)
                    print which_images, 'Completed'
            except Exception as e:
                print e
            cnt += 1
    print cnt


# Asserts whether final constructed input image and initial images are same, Runs payload tests for different capacities
def run_payloads_for_images_and_assert_final_image_equal_to_input(ip_img_path, tg_img_path, which_images, cnt):
    # No of characters in 10%, 20%, 40%, 50%, 60%, 80% of 512x512. Calculated using generate_percent_of_payload function
    payloads = [3277, 6554, 13108, 16384, 19661, 26215]
    payload_amount = ['payload_10', 'payload_20', 'payload_40', 'payload_50', 'payload_60', 'payload_80']
    # Create Directory
    path_prefix = 'test_results/' + str(cnt)
    if not os.path.exists(path_prefix):
        os.makedirs(path_prefix)
    ip_img = cv2.imread(ip_img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    tg_img = cv2.imread(tg_img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imwrite(path_prefix + '/input.png', ip_img)
    cv2.imwrite(path_prefix + '/target.png', tg_img)
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
        path_prefix = 'test_results/' + str(cnt) + '/' + payload_amount[i]
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        # For comparision

        """
        NOTE: Naming Nomenclature is incorrect
        - stego_img means the stego image
        - encoded_img means final image with message embedded
        """

        cv2.imwrite(path_prefix + '/target_img.png', tg_img)
        cv2.imwrite(path_prefix + '/stego_img.png', tf_img)
        cv2.imwrite(path_prefix + '/encoded_img.png', enc_tf_img)


def write_result_images_to_disk(filtered_list):
    img_path_prefix = 'resources/images/'
    cnt = 1
    for which_images in filtered_list:
        ip, tg = which_images.split("_")
        ip_img_path = img_path_prefix + str(ip) + '.pgm'
        tg_img_path = img_path_prefix + str(tg) + '.pgm'
        run_payloads_for_images_and_assert_final_image_equal_to_input(ip_img_path, tg_img_path, which_images, cnt)
        print cnt, '.', ' ', which_images, 'Done'
        cnt += 1


# List of 20 image pairings
# write_result_images_to_disk(['1_32', '2_12', '5_88', '20_47', '23_13', '24_26', '25_71', '28_15', '34_36', '37_21',
#                                '38_51', '40_31', '58_55', '72_39', '73_48', '75_45', '86_76', '96_35', '104_105',
#                                '106_102'])
