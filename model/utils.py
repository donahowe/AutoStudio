import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import inflect
import gradio as gr

p = inflect.engine()
user_error = gr.Error

# h, w
box_scale = (512, 512)
size = box_scale
size_h, size_w = size
print(f"Using box scale: {box_scale}")

def prepare_mid_image(mask_tensor_list_512, single_obj_img_list, bboxes, height, width, repeat_ind=0):
    mask_tensor_512 = mask_tensor_list_512[0]
    #m,n = mask_tensor_512.size()
    m,n = width, height
    new_mask_tensor = np.zeros((n, m)).astype(np.uint8)
    white_image = Image.new('RGB', (m, n), (0, 0, 0))

    tag = 0
    for image ,mask_tensor_512, box in zip(single_obj_img_list, mask_tensor_list_512, bboxes):
        tag += 1
        x_min, y_min, x_max, y_max = box[0], box[1], box[0]+box[2],  box[1]+box[3]
        box_center = [(x_min+x_max)/2, (y_min+y_max)/2]

        box_w = abs(x_max-x_min)
        box_h = abs(y_max-y_min)
        #m,n = mask_tensor_512.size()
        m,n = height, width
        abs_box_w = box_w*n
        abs_box_h = box_h*m
        abs_box_x_min = int(x_min*n)
        abs_box_y_min = int(y_min*m)

        min_mask_y, min_mask_x, max_mask_y, max_mask_x = find_bounding_box(mask_tensor_512)
        mask_w = abs(max_mask_x-min_mask_x)
        mask_h = abs(max_mask_y-min_mask_y)

        image_array = np.array(image)
        cropped_image_array = image_array[min_mask_y:max_mask_y, min_mask_x:max_mask_x, :]
        located_mask_tensor = mask_tensor_512[min_mask_y:max_mask_y,min_mask_x:max_mask_x]

        cropped_image = Image.fromarray(cropped_image_array)

        mask_array = np.where(located_mask_tensor, 255, 0).astype(np.uint8)
        mask_image = Image.fromarray(mask_array, mode='L')

        refactor = max(mask_w/abs_box_w, mask_h/abs_box_h)
        print("refactor: ",refactor)
        new_w = int(mask_w/refactor)
        new_h = int(mask_h/refactor)
        resize_img = cropped_image.resize((new_w,new_h))
        resize_mask = mask_image.resize((new_w,new_h))

        resize_mask_tensor = np.array(resize_mask)
        resize_img_tensor = np.array(resize_img)
        resize_mask_tensor[resize_mask_tensor > 0] = 255
        re_m, re_n = len(resize_mask_tensor), len(resize_mask_tensor[0])
        resize_mask_tensor_normalized = resize_mask_tensor / 255
        resize_img_tensor = resize_img_tensor * np.expand_dims(resize_mask_tensor_normalized.astype(np.uint8), axis=2)

        
        small_mask_tensor = Image.fromarray(resize_mask_tensor, mode='L')
        resize_factor = 1 #parame
        new_mask_resized = small_mask_tensor.resize((int(re_n * resize_factor), int(re_m * resize_factor)), Image.BOX)
        final_mask = Image.new('L', (re_n, re_m), color=0)
        x_offset = (re_n - int(re_n * resize_factor)) // 2
        y_offset = (re_m - int(re_m * resize_factor)) // 2
        final_mask.paste(new_mask_resized, (x_offset, y_offset))
        
        resize_mask_tensor = np.array(final_mask)
        resize_mask_tensor[resize_mask_tensor > 0] = 255
        
        img_cover_tensor = ~new_mask_tensor.copy()
        img_cover_tensor = img_cover_tensor / 255 
        img_cover_tensor2 = new_mask_tensor.copy()
        img_cover_tensor2 = img_cover_tensor2 / 255 
        
        white_array = np.array(white_image)

        crop_m = len(white_array[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n, :])
        cop_n = len(white_array[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n, :][0])
        resize_img_tensor = resize_img_tensor[:crop_m, :cop_n]
        resize_mask_tensor = resize_mask_tensor[:crop_m, :cop_n]

        white_array[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n, :] = resize_img_tensor
        white_array = white_array * np.expand_dims(img_cover_tensor.astype(np.uint8), axis=2)

        origin_white_array = np.array(white_image)
        final_array = white_array + (origin_white_array * np.expand_dims(img_cover_tensor2.astype(np.uint8), axis=2))

        white_image = Image.fromarray(final_array)

        #deal masks
        new_mask_tensor[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n] = new_mask_tensor[abs_box_y_min : abs_box_y_min + re_m, abs_box_x_min : abs_box_x_min + re_n] + resize_mask_tensor
        new_mask_tensor[new_mask_tensor>255] = 255
        new_mask = Image.fromarray(~new_mask_tensor, mode='L')
        new_mask_T = Image.fromarray(new_mask_tensor, mode='L')
        
    print("Down Inpainting Rreperation")
    return new_mask, white_image

def find_bounding_box(tensor):
    true_indices = torch.nonzero(tensor)
    if true_indices.shape[0] == 0:
        return None
    min_row = true_indices[:, 0].min().item()
    max_row = true_indices[:, 0].max().item()
    min_col = true_indices[:, 1].min().item()
    max_col = true_indices[:, 1].max().item()

    return min_row, min_col, max_row, max_col


def get_global_prompt(prompt_book):

    obj_counts = {}
    for item in prompt_book['obj_ids']:
        if item in obj_counts:
            obj_counts[item] += 1
        else:
            obj_counts[item] = 1

    character_prompts = ""

    for i in range(len(prompt_book['obj_ids'])):
        if str(prompt_book['obj_ids'][i]).find("-") != -1:
            continue
        character_prompt = prompt_book['gen_boxes'][i][0]
        if obj_counts[prompt_book['obj_ids'][i]] == -1:
            continue

        elif obj_counts[prompt_book['obj_ids'][i]] == 1:
            character_prompts = character_prompts + " " + character_prompt + ","

        elif obj_counts[prompt_book['obj_ids'][i]] > 1:
            character_prompts = character_prompts + " " + str(obj_counts[prompt_book['obj_ids'][i]]) + character_prompt + ","
            obj_counts[prompt_book['obj_ids'][i]] = -1
        
    global_prompt = f"{prompt_book['bg_prompt']} with{character_prompts}"
    prompt_book['global_prompt'] = global_prompt
    return prompt_book

def show_boxes(size, gen_boxes, bg_prompt=None, neg_prompt=None, ind=None, show=False, save_dir=None):
    if len(gen_boxes) == 0:
        return
    
    if isinstance(gen_boxes[0], dict):
        anns = [{'name': gen_box['name'], 'bbox': gen_box['bounding_box']}
                for gen_box in gen_boxes]
    else:
        anns = [{'name': gen_box[0], 'bbox': gen_box[1]} for gen_box in gen_boxes]

    # White background (to allow line to show on the edge)
    I = np.ones((size[0]+4, size[1]+4, 3), dtype=np.uint8) * 255
    
    plt.figure()
    plt.imshow(I)
    plt.axis('off')

    if bg_prompt is not None:
        ax = plt.gca()
        ax.text(0, 0, bg_prompt + f"(Neg: {neg_prompt})" if neg_prompt else bg_prompt, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        c = (np.zeros((1, 3)))
        [bbox_x, bbox_y, bbox_w, bbox_h] = (0, 0, size[1], size[0])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons = [Polygon(np_poly)]
        color = [c]
        p = PatchCollection(polygons, facecolor='none',
                            edgecolors=color, linewidths=2)
        ax.add_collection(p)

    draw_boxes(anns)
    if show:
        plt.show()
    else:
        print("Saved boxes visualizations to", f"{save_dir}/boxes.png", f"ind: {ind}")
        if ind is not None:
            plt.savefig(f"{save_dir}/boxes_{ind}.png")
        plt.savefig(f"{save_dir}/boxes.png")

def show_image(image, save_prefix="", ind=None, time_step=None, save_dir=None):

    ind = f"{ind}" if ind is not None else ""
    path = f"{save_dir}/{save_prefix}{ind}_{time_step}step.png"
    
    print(f"Saved to {path}")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    print(path)
    print(image)
    image.save(path)

def draw_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        name = ann['name'] if 'name' in ann else str(ann['category_id'])
        ax.text(bbox_x, bbox_y, name, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    p = PatchCollection(polygons, facecolor='none',
                        edgecolors=color, linewidths=2)
    ax.add_collection(p)
    

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map

def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images

def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):
    
    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator