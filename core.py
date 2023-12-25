from camelot.core import Cell,Table
from functools import reduce
import numpy as np
import pandas as pd
from .utils.image_processing import resize

class Cell(Cell):
    def __init__(self,x1, y1, x2, y2):
        super().__init__(x1, y1, x2, y2)
        self.r_x1 = int(self.x1)
        self.r_y1 = int(self.y1)
        self.r_x2 = int(self.x2)
        self.r_y2 = int(self.y2)
        self.cropped_range = [self.r_x1,self.r_y1,self.r_x2,self.r_y2]
        self.coors = np.array([[self.r_x1,self.r_y1],
                               [self.r_x2,self.r_y1],
                               [self.r_x2,self.r_y2],
                               [self.r_x1,self.r_y2]],np.int32)
        self.score = None

class Table(Table):
    def __init__(self,cols,rows):
        super().__init__(cols,rows)
        self.cols = cols
        self.rows = rows
        self.cells = [[Cell(c[0], r[1], c[1], r[0]) for c in cols] for r in rows][::-1]
        self.shape = np.array(self.cells).shape
        self.all_cells = reduce(lambda x,y:x+y,self.cells)
        self.conflict_merged_cells = []
        
    @property
    def avg_score(self):
        """calculate average recognition confidence score of the whole table"""
        scores = [cell.score for cell in self.all_cells if pd.notnull(cell.score)]
        if scores:
            avg_score = sum(scores)/len(scores)
        else:
            avg_score = np.nan
        return avg_score
    
    def set_edges(self,joint_tol=3):
        """
        Set cells' edges based on cell object.
        This is an alternative to camelot.core.table.set_edges
        check if the cell's side lies on the segment.
        
        """
        for cell in self.all_cells:
            #find the horizontal segment that is close to the cell's top edge
            midpoint_x = (cell.x1 + cell.x2)/2
            midpoint_y = (cell.y1 + cell.y2)/2
            cell_t_on_h = [h for h in self.h_s if np.isclose(cell.y1, h[1], atol=joint_tol) and h[0] <= midpoint_x <= h[2]]
            cell_b_on_h = [h for h in self.h_s if np.isclose(cell.y2, h[1], atol=joint_tol) and h[0] <= midpoint_x <= h[2]]
            cell_l_on_v = [v for v in self.v_s if np.isclose(cell.x1, v[0], atol=joint_tol) and v[1] <= midpoint_y <= v[3]]
            cell_r_on_v = [v for v in self.v_s if np.isclose(cell.x2, v[0], atol=joint_tol) and v[1] <= midpoint_y <= v[3]]
            
            if cell_t_on_h:
                #check if the midpoint of the cell's bottom edge lies on the horizontal segment that close to the cell's bottom edge
                cell.top = True
            if cell_b_on_h:
                cell.bottom = True
            if cell_l_on_v:
                cell.left = True
            if cell_r_on_v:
                cell.right = True
                    
    def set_cropped_range(self):
        """
        set the range of image to be cropped to do the recognization : [x1,y1,x2,y2]
        original cropped range is set to be the cell's range.
        check if cells are merged from top to bottom, and then from left to right by checking the cell's edge. 
        If yes, expand the cell's cropped range to the cell that is merged.
        
        example:
      (0,0)   (2,0)    (4,0)
        -----------------
        |   A   |   B   |    if cell A's right edge is True, nothing happens. cell A's cropped range is [0,0,2,2]
        -----------------
      (0,2)   (2,2)    (4,2)
      
        
      (0,0)   (2,0)    (4,0)
        -----------------
        |   A   |   B   |    if cell A's right edge is False, cell A's cropped range becomes [0,0,4,2], and cell B's cropped range will be set to be None.
        -----------------
      (0,2)   (2,2)    (4,2)
        """
        for y, cols in enumerate(self.cells):
            for x, row in enumerate(cols):
                if self.cells[y][x].bound == 4 or not self.cells[y][x].cropped_range:
                    continue
                
                current_x = x
                while not self.cells[y][current_x].right:
                    if current_x == self.shape[1]-1:
                        break
                    elif self.cells[y][current_x+1].cropped_range:
                        self.cells[y][x].cropped_range[-2] = self.cells[y][current_x+1].cropped_range[-2]
                        self.cells[y][current_x+1].cropped_range = None
                        current_x+=1
                    else:
                        self.conflict_merged_cells.append((y,current_x+1))
                        current_x+=1
                        pass
                    
                current_y = y
                while not self.cells[current_y][x].bottom:
                    if current_y == self.shape[0]-1:
                        break
                    elif self.cells[current_y+1][x].cropped_range:
                        self.cells[y][x].cropped_range[-1] = self.cells[current_y+1][x].cropped_range[-1]
                        self.cells[current_y+1][x].cropped_range = None
                        current_y+=1
                    else:
                        self.conflict_merged_cells.append((current_y+1,x))
                        current_y+=1
                        pass
                        
    def generate_cropped_img(self,img,size=1.5):
        """img[y1:y2,x1:x2]"""
        
        for cell in self.all_cells:
            if cell.cropped_range:
                cell.cropped_img = img[cell.cropped_range[1]:cell.cropped_range[3],cell.cropped_range[0]:cell.cropped_range[2]]
                cell.cropped_img = resize(cell.cropped_img,scale=size)
                
    def reset_edges(self):
        for cell in self.all_cells:
            cell.top = cell.bottom = cell.left = cell.right = False