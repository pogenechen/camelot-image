from .utils.image_processing import *
from .utils.utils import *
from .core import *
from .utils.oc import oc

class lattice():
    def __init__(self,
                 lang='en',
                 pdf_path='',
                 page='',
                 img_idx='',
                 image_path='',
                 line_tol=5,
                 joint_tol=0.002,
                 score_thresh=0.7,
                 **kwargs):
        
        super().__init__(pdf_path,
                         page,
                         img_idx,
                         image_path,
                         line_tol,
                         joint_tol,
                         **kwargs)
        
        self.lang = lang
        self.pdf_path = pdf_path
        self.page = page
        self.img_idx = img_idx
        self.image_path = image_path
        self.line_tol = line_tol
        self.joint_tol = joint_tol
        self.score_thresh = score_thresh
        self.thickness = kwargs.get('border_thickness',3)
        
        if all((self.pdf_path,self.page)):
            if self.image_path:
                raise Exception('Can only specify either pdf or image.')
            else:
                self.doc = open_pdf(self.pdf_path)
                self.img_dict = get_pdf_img(self.doc)
                self.image = self.img_dict[f'page {self.page}'][f'image {self.img_idx}']
            
        elif not any((self.doc,self.img_idx,self.page)):
            if self.image_path:
                self.image = cv2.imread(self.image_path)
                if len(self.image.shape) == 3 and self.image.shape[-1] != 1:
                    self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
            else:
                raise Exception('Must specify pdf or an independent image.')
        else:
            raise Exception('Must specify pdf or an independent image.')
    
    def _generate_table_bbox(self):
        """
        detect tables with borders and their joints, and all the horizontal and vertical lines.
        """
        o = oc(self.image)
        self.image = o.correct_text_skewness()
        thresh = self.image.copy()
        thresh = draw_straight_lines(thresh,self.thickness)
        self.hor_mask, hor_lines, self.ver_mask, ver_lines = find_lines(thresh)
        self.horizontal_segments = sorted(hor_lines,key=lambda x:x[1])
        self.vertical_segments = sorted(ver_lines,key=lambda x:x[0])
        contours = find_contours(self.ver_mask,self.hor_mask)
        self.table_bbox = find_joints(contours,self.ver_mask,self.hor_mask)
        
    def recognition_task(self,params):
        """recognize texts in the cropped image of the cell
        params: (cell, ocr recognizer) 
        """
        try:
            cell,recognizer = params
            recog_result = recognizer.ocr(cell.cropped_img)
            if recog_result:
                text = '\n'.join([i['text'] for i in recog_result])
                score = [i['score'] if i['score'] else 0 for i in recog_result]
                score = sum(score)/len(score)
            else:
                text = ''
                score = None
            cell.text = text
            cell.score = score
        except:
            print(traceback.format_exc())
        
    def _recognize_each_cell(self,table):
        """expedite recognition task with threading"""
        recognizer = ocr(self.lang)
        task_lst = [(cell,recognizer) for cell in table.all_cells if cell.cropped_range]
        with ThreadPoolExecutor() as executor:
            executor.map(self.recognition_task,task_lst)
            
    def _generate_columns_and_rows(self,tk):
        """
    (x1,y1)   .....     (x5,y1)
        -----------------
        |   |   |   |   |
        -----------------
        |   |   |   |   |
        -----------------
    (x1,y3)   .....     (x5,y3)    
    
        cols: [(x1,x2),(x2,x3),(x3,x4),(x4,x5)]
        rows: [(y3,y2),(y2,y1)]
        """
        v_s, h_s = segments_in_bbox(tk, self.vertical_segments, self.horizontal_segments,tol=self.line_tol)
        cols, rows = zip(*self.table_bbox[tk])
        cols, rows = list(cols), list(rows)
        # sort horizontal and vertical segments
        cols = merge_close_lines(sorted(cols), line_tol=self.line_tol)
        rows = merge_close_lines(sorted(rows, reverse=True), line_tol=self.line_tol)
        cols = [tk[0]] + cols[1:-1] + [tk[2]]
        rows = [tk[3]] + rows[1:-1] + [tk[1]]
        # make grid using x and y coord of shortlisted rows and cols
        cols = [(cols[i], cols[i + 1]) for i in range(0, len(cols) - 1)]
        rows = [(rows[i], rows[i + 1]) for i in range(0, len(rows) - 1)]
        return cols, rows, v_s, h_s
    
    def check_border_existence(self):
        self._generate_table_bbox()
        if not self.table_bbox:
            return False
        else:
            return True
        
    def extract_tables(self):
        
        check = self.check_border_existence()
        if not check:
            raise ValueError('no table with borders detected.')
        try_cnt = 0
        while True:
            if try_cnt == 4:
                print('Recognition Failed.')
                break
            _tables = []
            for table_idx, tk in enumerate(sorted(self.table_bbox.keys(), key=lambda x: x[1], reverse=True)):
                cols, rows, v_s, h_s = self._generate_columns_and_rows(tk)
                table = Table(cols,rows)
                table.v_s = v_s
                table.h_s = h_s
                
                #set edges for all cells --> set cropped range for all cells --> set cropped image based on cropped range --> recognized text in the cropped images.
                #tolerance of cell's side and segment is set to be a proportion of image's longer side
                joint_rtol = self.joint_tol*(max(self.image.shape[:2]))
                table.set_edges(joint_tol=joint_rtol)
                table.set_cropped_range()
                table.generate_cropped_img(self.image)
                self._recognize_each_cell(table)
                    
                table.df = pd.DataFrame(table.data)
                table.flavor = "lattice"
                _tables.append(table)
            
            #if average confidence score is lower than 0.7, then assume that the image is wrongly oriented.
            # rotate the image and run agian
            table_scores = [t.avg_score for t in _tables]
            if all([pd.notnull(score) for score in table_scores]):
                avg = sum(table_scores)/len(table_scores)
                if avg > self.score_thresh:
                    break
                else:
                    self.image = rotate_90(self.image)
                    print(f'average score is only {avg:.1%}.  Rotate the image by 90 degrees and try again')
                    self._generate_table_bbox()
                    try_cnt+=1
            else:
                pass
        return _tables