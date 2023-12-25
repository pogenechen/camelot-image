import numpy as np
import cv2
import statistics

def resize(img,scale:float=None,square_length:int=None,inter=cv2.INTER_CUBIC):
    if not scale and not square_length:
        raise Exception('Please specify a resizing method, either "scale" or "square_length."')
    if scale != None and square_length != None:
        raise Exception('Can only choose one resizing method.')
    if scale:
        img = cv2.resize(img,None,fx=scale,fy=scale,interpolation=inter)
    elif square_length:
        img = cv2.resize(img,(square_length,square_length),interpolation=inter)
    return img

def show(img,scale:float=0.3,square_length:int=None):
    img_ = img.copy()
    img_ = resize(img_,scale,square_length)
    # cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image',img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rotate(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT)

def rotate_90(img):
    h,w = img.shape[:2]
    #resize the image to a square and then resize it back to original size after rotated so that the image won't be distorted.
    img = cv2.resize(img,(2000,2000),interpolation=cv2.INTER_CUBIC)
    
    x,y = img.shape[:2]
    center = (x//2,y//2)
    M = cv2.getRotationMatrix2D(center, -90, 1.0)
    rotated = cv2.warpAffine(img, M, (y,x),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    img = cv2.resize(rotated,(h,w),interpolation=cv2.INTER_CUBIC)
    return img

def interchange(img):
    #interchange gray scale and BGR scale
    if len(img.shape) == 3 and img.shape[-1] != 1:
        return cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)

    elif len(img.shape) == 2 or img.shape[-1] == 1:
        return cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)

def draw(img,coors,**kwargs):
    if len(img.shape) == 2 or img.shape[-1] == 1:
        color_img = interchange(img.copy())
    else:
        color_img = img.copy()
    
    color = kwargs.get('color',(0,0,255))
    thickness = kwargs.get('thickness',2)
    scale = kwargs.get('scale',.3)
    type_ = kwargs.get('type','polylines')
    
    for coor in coors:
        if type_ == 'polylines':
            coor = coor.reshape(-1,1,2)
            cv2.polylines(color_img, [coor], True, color, thickness)
        elif type_== 'circle':
            cv2.circle(color_img,coor,thickness,color,-1)
        elif type_ == 'ver-line':
            cv2.line(color_img,(coor[0],coor[1]),(coor[0],coor[3]),color,thickness)
        elif type_ == 'hor-line':
            cv2.line(color_img,(coor[0],coor[1]),(coor[2],coor[1]),color,thickness)    
        else:
            raise Exception('"type" should be either "polylines" or "circle.')
    show(color_img,scale=scale)
    
        
def bgr_process(img,upper = [100,100,100]):
    #add a filter on BGR image to remove colors that are not black.
    img = img.copy()
    if len(img.shape) == 3 and img.shape[-1] != 1:
        lower_black = np.array([0,0,0])
        upper_black = np.array(upper)

        mask = cv2.inRange(img, lower_black, upper_black)
        res = cv2.bitwise_and(img, img, mask=mask)
        res[mask==0] = [255,255,255]
        res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        return res
    else:
        return img
    
def pix_to_image(pix):
    bytes_ = np.frombuffer(pix.samples, dtype=np.uint8)
    img = bytes_.reshape(pix.height, pix.width, pix.n)
    return img

def draw_straight_lines(img,horizontal=True,vertical=True,thickness=5):
    img = img.copy()
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernels = []
    if horizontal:
        kernels.append(cv2.getStructuringElement(cv2.MORPH_RECT, (25,1)))
    if vertical:
        kernels.append(cv2.getStructuringElement(cv2.MORPH_RECT, (1,25)))
    assert kernels, 'should choose the type(s) of straight line you want to remove.'
    for kernel in kernels:
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            cv2.drawContours(thresh, [c], -1, (255,255,255), thickness)
    return thresh

def orientation_correction(img,method='each-obj',display=False):
    img = img.copy()
    ori = img.copy()
    if len(img.shape) == 3 and img.shape[-1] != 1:
        ori_color = img.copy()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        ori_color = cv2.cvtColor(img.copy(),cv2.COLOR_GRAY2BGR)
        
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
    img = cv2.medianBlur(img,3)

    if method == 'each-obj':
        kernel_size=(5,100)
        kernel = np.ones(kernel_size, np.uint8)
        img = cv2.erode(img, kernel, iterations = 1)
        img = cv2.dilate(img, kernel, iterations = 1)
        contours, hiearachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        angles = []

        for idx,cnt in enumerate(contours):
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            box = cv2.boxPoints(rect)
            box = np.int0(cv2.boxPoints(rect))
            angles.append(angle)
            cv2.drawContours(ori_color, [box], 0, (0, 0, 255), 2)

        angles = [0 if i%90==0 else i for i in angles]
        angles = [i for i in angles if i/45 not in (1,3,5,7)]
        data = np.array(angles)
        q1 = np.percentile(data, 50)
        q3 = np.percentile(data, 60)
        iqr = q3 - q1
        transform = data[(data >= (q1 - 1.5 * iqr)) & (data <= (q3 + 1.5 * iqr))]
        transform = [-i+90 if i>45 else -i for i in transform]
        angle = sum(transform)/len(transform)

    elif method == 'whole-page':
        img = cv2.medianBlur(img,3)
        contours, hiearachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mode = statistics.mode([i[-1] for i in hiearachy[0]])
        eles = [contours[i] for i,j in enumerate(hiearachy[0]) if j[-1]==mode]
        coords = []
        for ele in eles:
            for coor in ele:
                coords.append(list(coor[0]))
        coords = np.array(coords)
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = 90 - abs(angle) if abs(angle) > 45 else -angle
        if display:
            box = cv2.boxPoints(rect)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(ori_color, [box], 0, (0, 0, 255), 2)
            
    elif method == 'largest-box':
        height, width = img.shape[:2]
        kernel_size = (int(height*0.02), int(width*0.05))
        _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
        img = cv2.medianBlur(img,3)
        kernel = np.ones(kernel_size, np.uint8)
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations=1)
        img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel, iterations=1)
        contours, hiearachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_area = img.shape[0]*img.shape[1]
        boxes=[]
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(cv2.boxPoints(rect))
            if np.isclose(cv2.contourArea(box),img_area,rtol=0.03):
                continue
            boxes.append((box,rect[-1]))

        largest_box = sorted(boxes,key=lambda x:cv2.contourArea(x[0]),reverse=True)[0]
        angle = largest_box[-1]
        angle = 90 - abs(angle) if abs(angle) > 45 else -angle
    
    if display == 'normal':
        show(rotate(ori_color,-angle),scale=.3)
    elif display == 'inverse':
        show(rotate(cv2.bitwise_not(ori_color),-angle),scale=.3)
    return rotate(ori,-angle)

#找出文字區域的輪廓
def get_rect(img,kernel_size=(5, 100),display=False):
    """
    Find out contours of text boxes.
    
    Returns
    -------
    rect_list: list. list of tuple:(y_min,y_max,x_min,x_max)
    """
    img = img.copy()
    original = img.copy()
    if len(img.shape) == 3 and img.shape[-1] != 1:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        original = cv2.cvtColor(original,cv2.COLOR_GRAY2BGR)
    rect_lst = []
    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)
    img = cv2.medianBlur(img,3)
    kernel = np.ones(kernel_size, np.uint8)
    img = cv2.erode(img, kernel, iterations = 1)
    img = cv2.dilate(img, kernel, iterations = 1)
    contours, hiearachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mode = statistics.mode([i[-1] for i in hiearachy[0]])

    for idx,cnt in enumerate(contours):
        #抓取最多個數的那個層級的輪廓(同個父輪廓下最多個數的層級)
        if hiearachy[0][idx][-1] != mode:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        rect_lst.append((y,y+h,x,x+w))
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 3)

    if display:
        show(original,scale=.2)
    return rect_lst


#===============================| camelot.image_processing |==================================
#revise to fit image coordinates.
def adaptive_threshold(imagename, process_background=False, blocksize=15, c=-2):
    """Thresholds an image using OpenCV's adaptiveThreshold.

    Parameters
    ----------
    imagename : string
        Path to image file.
    process_background : bool, optional (default: False)
        Whether or not to process lines that are in background.
    blocksize : int, optional (default: 15)
        Size of a pixel neighborhood that is used to calculate a
        threshold value for the pixel: 3, 5, 7, and so on.

        For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.
    c : int, optional (default: -2)
        Constant subtracted from the mean or weighted mean.
        Normally, it is positive but may be zero or negative as well.

        For more information, refer `OpenCV's adaptiveThreshold <https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html#adaptivethreshold>`_.

    Returns
    -------
    img : object
        numpy.ndarray representing the original image.
    threshold : object
        numpy.ndarray representing the thresholded image.

    """
    img = cv2.imread(imagename) if isinstance(imagename,str) else imagename
    if len(img.shape) == 3 and img.shape[-1] != 1:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    if process_background:
        threshold = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blocksize, c
        )
    else:
        threshold = cv2.adaptiveThreshold(
            np.invert(gray),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blocksize,
            c,
        )
    return img, threshold


def find_lines(img,horizontal=True,vertical=True,kernel_size=25):
    img = img.copy()
    assert any([horizontal,vertical])
    thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernels = {}
    if horizontal:
        kernels['horizontal'] = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size,1))
    if vertical:
        kernels['vertical'] = cv2.getStructuringElement(cv2.MORPH_RECT, (1,kernel_size))
    assert kernels, 'should choose the type(s) of straight line you want to remove.'
    
    lines = {}
    for direction, kernel in kernels.items():
        detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        line_coors = []
        for cnt in cnts:
            cnt = cnt.reshape(-1,2)
            if direction == 'horizontal':
                x1 = int(min(cnt[:,0]))
                x2 = int(max(cnt[:,0]))
                y1 = y2 = int((min(cnt[:,1]) + max(cnt[:,1]))/2)
                line_coors.append((x1,y1,x2,y2))
            elif direction == 'vertical':
                x1 = x2 = int((min(cnt[:,0]) + max(cnt[:,0]))/2)
                y1 = int(min(cnt[:,1]))
                y2 = int(max(cnt[:,1]))
                line_coors.append((x1,y1,x2,y2))
        lines[direction] = {'mask':detected_lines,'lines':line_coors}
    if all([horizontal, vertical]):
        return lines['horizontal']['mask'], lines['horizontal']['lines'], lines['vertical']['mask'], lines['vertical']['lines']
    elif horizontal:
        return lines['horizontal']['mask'], lines['horizontal']['lines']
    else:
        return lines['vertical']['mask'], lines['vertical']['lines']

def find_contours(vertical, horizontal):
    """Finds table boundaries using OpenCV's findContours.

    Parameters
    ----------
    vertical : object
        numpy.ndarray representing pixels where vertical lines lie.
    horizontal : object
        numpy.ndarray representing pixels where horizontal lines lie.

    Returns
    -------
    cont : list
        List of tuples representing table boundaries. Each tuple is of
        the form (x, y, w, h) where (x, y) -> left-top, w -> width and
        h -> height in image coordinate space.

    """
    mask = vertical + horizontal

    try:
        __, contours, hier = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    except ValueError:
        # for opencv backward compatibility
        contours, hier = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    #assume table contours have more than one child contour
    parent_idx = hier[0][:,3]
    more_than_1_child_idx = [idx for idx in set(parent_idx) if parent_idx.tolist().count(idx)>1 and idx!=-1]
    contours = [contours[idx] for idx in more_than_1_child_idx]
    # sort in reverse based on contour area and use first 10 contours
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    cont = []
    for c in contours:
        c_poly = cv2.approxPolyDP(c, 3, True)
        x, y, w, h = cv2.boundingRect(c_poly)
        cont.append((x, y, w, h))
    return cont


def find_joints(contours, vertical, horizontal):
    """Finds joints/intersections present inside each table boundary.

    Parameters
    ----------
    contours : list
        List of tuples representing table boundaries. Each tuple is of
        the form (x, y, w, h) where (x, y) -> left-top, w -> width and
        h -> height in image coordinate space.
    vertical : object
        numpy.ndarray representing pixels where vertical lines lie.
    horizontal : object
        numpy.ndarray representing pixels where horizontal lines lie.

    Returns
    -------
    tables : dict
        Dict with table boundaries as keys and list of intersections
        in that boundary as their value.
        Keys are of the form (x1, y1, x2, y2) where (x1, y1) -> lb
        and (x2, y2) -> rt in image coordinate space.

    """
    joints = np.multiply(vertical, horizontal)
    tables = {}
    for c in contours:
        x, y, w, h = c
        roi = joints[y : y + h, x : x + w]
        try:
            __, jc, __ = cv2.findContours(
                roi.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
        except ValueError:
            # for opencv backward compatibility
            jc, __ = cv2.findContours(
                roi.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
        # if len(jc) <= 4:  # remove contours with less than 4 joints
        #     continue
        joint_coords = []
        for j in jc:
            jx, jy, jw, jh = cv2.boundingRect(j)
            c1, c2 = x + (2 * jx + jw) // 2, y + (2 * jy + jh) // 2
            joint_coords.append((c1, c2))
        tables[(x, y, x + w, y + h)] = joint_coords

    return tables
