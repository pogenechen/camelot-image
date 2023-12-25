from .image_processing import *
from cnocr import CnOcr
import time, traceback
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import fitz

def open_pdf(pdf):
    return fitz.open(pdf)

def get_pdf_img(doc):
    img_dict = defaultdict(dict)
    for idx1, page in enumerate(doc):
        for idx2,img in enumerate(doc.get_page_images(page)):
            try:
                arrayed = pix_to_image(fitz.Pixmap(doc,img[0]))
                img_dict[f"page {idx1+1}"].update({f"image {idx2+1}":arrayed})
            except Exception as e:
                print(f"page: {idx1}, # of pics: {idx2}\n{e}")
    return img_dict

def ocr(lang):
    model_mapping = {
        'infer':{'en':'en_PP-OCRv3',
                'ch':'ch_PP-OCRv3'},
        'detect':{'en':'en_PP-OCRv3_det',
                'ch':'ch_PP-OCRv3_det'}
        }
    _ocr = CnOcr(det_model_name=model_mapping['detect'].get(lang), 
                rec_model_name=model_mapping['infer'].get(lang))
    return _ocr

def identify_rect_task(params):
    try:
        img,margin,text_rect,result_lst,ocr = params
        img=img.copy()
        y_min,y_max,x_min,x_max = text_rect
        coordinates = np.array([[x_min,y_min],
                                [x_max,y_min],
                                [x_max,y_max],
                                [x_min,y_max]])
        border = [margin]*4
        cropped_img = img[y_min:y_max,x_min:x_max]
        cropped_img = cv2.copyMakeBorder(cropped_img, *border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # cropped_img = cv2_rotation(cropped_img)
        result = ocr.ocr(cropped_img)

        # 長方形有些框得太大，影響座標 -> 用辨識的position去調整長方形
        # 只調整x座標，不調整y座標，保持文字同高
        if len(result) > 1:
            for r in result:
                mapped_x_min = min(r['position'][:,0])*(x_max-x_min)/cropped_img.shape[1]+x_min
                mapped_x_max = max(r['position'][:,0])*(x_max-x_min)/cropped_img.shape[1]+x_min
                mapped_coor = np.array([[mapped_x_min,y_min],[mapped_x_max,y_min],
                                        [mapped_x_max,y_max],[mapped_x_min,y_max]])
                result_lst+=[(r['text'],mapped_coor,r['score'])]
        else:
            result_lst+=[(i['text'],coordinates,i['score']) for i in result]
    except:
        print(traceback.format_exc())
        
def identify_rect(img,ocr,resize_scale=1.5,margin=15):
    start = time.time()
    img = img.copy()
    img = bgr_process(img)
    img = resize(img,scale=resize_scale)
    # img = pdf_recognizer.remove_straight_lines(img)

    cnt = 0

    while True:
        score = 0
        img = orientation_correction(img)
        text_rect_list = get_rect(img)
        result_lst = []
        task_list = [(img,margin,text_rect,result_lst,ocr) for text_rect in text_rect_list]
        if cnt == 4:
            print(f'\r轉了一圈了都不行  花費時間:{time.time()-start:.1f}秒')
            break
        with ThreadPoolExecutor() as executor:
            executor.map(identify_rect_task,task_list)
        
        only_12_ch = len([t for t,_,_ in result_lst if len(t) < 3])
        
        if len(result_lst) > 0:
            if only_12_ch/len(result_lst) >= 0.8:
                score = 0
            else:
                score_lst = [score for text,_,score in result_lst if len(text)>2 and not pd.np.isnan(score)]
                score = sum(score_lst)/len(score_lst)
        else:
            score = 0
        if score < 0.6:
            cnt+=1
            img = rotate_90(img)
            img = orientation_correction(img)
            print(f'\nscore只有{score:.1%}  轉個90度試試')
        else:
            print(f'\r辨識完成。 confidence score: {score:.1%}  花費時間:{time.time()-start:.1f}秒')
            break
    result_lst = [(text,coor) for text,coor,sc in result_lst if sc>0.4]
    return img, result_lst, score

def segments_in_bbox(bbox, v_segments, h_segments, tol):
    """Returns all line segments present inside a bounding box.

    Parameters
    ----------
    bbox : tuple
        Tuple (x1, y1, x2, y2) representing a bounding box where
        (x1, y1) -> lb and (x2, y2) -> rt in PDFMiner coordinate
        space.
    v_segments : list
        List of vertical line segments.
    h_segments : list
        List of vertical horizontal segments.

    Returns
    -------
    v_s : list
        List of vertical line segments that lie inside table.
    h_s : list
        List of horizontal line segments that lie inside table.

    """
    lb = (bbox[0], bbox[1])
    rt = (bbox[2], bbox[3])
    v_s = [
        v
        for v in v_segments
        if v[1] > lb[1] - tol and v[3] < rt[1] + tol and lb[0] - tol <= v[0] <= rt[0] + tol
    ]
    h_s = [
        h
        for h in h_segments
        if h[0] > lb[0] - tol and h[2] < rt[0] + tol and lb[1] - tol <= h[1] <= rt[1] + tol
    ]
    return v_s, h_s

def merge_close_lines(ar, line_tol=2, method='ma'):
    """Merges lines which are within a tolerance by calculating a
    moving mean, based on their x or y axis projections.

    Parameters
    ----------
    ar : list
    line_tol : int, optional (default: 2)

    Returns
    -------
    ret : list

    """
    ret = []
    for a in ar:
        if not ret:
            ret.append(a)
        else:
            temp = ret[-1]
            if np.isclose(temp, a, atol=line_tol):
                if method == 'ma':
                    temp = (temp + a) / 2.0
                elif method == 'last':
                    temp = a
                ret[-1] = temp
            else:
                ret.append(a)
    return ret