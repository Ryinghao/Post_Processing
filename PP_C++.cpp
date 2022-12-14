#include<iostream>
#include<unordered_map>
#include<map>
#include<vector>
#include<math.h>
#include<NMS.cpp>
using namespace std;

struct detection{
    vector<vector<float>>boxes;
    vector<float>scores;
    vector<int>labels;
};

int num_classes = 264;
const int N = 1;
const int H = 100;
const int W = 100;
const int image_size = 800;
const int num_images=1;
const int score_thresh = 0.05;
vector<float> process_tensor(vector<float>&in_ptr, int in_depth, int in_height, int in_width){
    int s = in_ptr.size();
    vector<float>out_ptr(s);
    int count = 0;
    int A = in_depth/num_classes;

    for (int n = 0; n < N; ++n) {
      for (int x = 0; x < A; ++x) {
        for (int c = 0; c < num_classes; ++c) {
          for (int h = 0; h < in_height; ++h) {
            for (int w = 0; w < in_width; ++w) {
               int in_idx = n * (in_height * in_width * in_depth) + x * (num_classes * in_height * in_width) + c * (in_height * in_width) + h * (in_width) + w;
               int out_idx = n * (in_height * in_width * in_depth) + h * (in_width * in_depth) + w * (in_depth) + x * (num_classes) + c;
               out_ptr[out_idx] = in_ptr[in_idx];
            }
          }
        }
      }
    }
    return out_ptr;
}

void load_anchors(vector<vector<vector<vector<float>>>>&anchors_for_images){

}

vector <float> sigmoid (const vector<float>& m1, int img_num, int VECTOR_SIZE) {
    int start = img_num*VECTOR_SIZE;
    int end = (img_num+1)*VECTOR_SIZE;

    vector <float> output (VECTOR_SIZE);
    
    for( unsigned t = start; t != end; ++t) {
        output[t] = 1 / (1 + exp(-m1[t]));
    }
    
    return output;
}



vector<int> flatten(vector<int>in){
    return in;
}



vector<vector<float>> box_to_picture_clip(vector<int>idxs, vector<float>&bbox_regression_level, int len_dim, int img_num, 
vector<vector<float>>&allboxes, int image_size){
     
     float bbox_xform_clip = 4.135166556742356; //it needs a function later handle
     vector<int>weights = {1,1,1,1};
     int start = img_num*len_dim;
     int end = (img_num+1)*len_dim;
     int sz = idxs.size();
     vector<vector<float>>rel_codes;
     vector<vector<float>>boxes(sz, vector<float>(4));
     vector<vector<float>>res(sz, vector<float>(4));
     for(int i=0; i<sz;i++){
        int t = idxs[i];
        boxes[i] = allboxes[t];// is this are as the order with the original progrm? 
        vector<float>tmp(bbox_regression_level.begin()+start+t*4, bbox_regression_level.begin()+start+(t+1)*4);
        rel_codes.push_back(tmp);
        //please verify it later
     }

     vector<float>widths(sz);
     vector<float>heights(sz);
     vector<float>ctr_x(sz);
     vector<float>ctr_y(sz);
     for(int i=0; i<sz; i++){
        widths[i] = boxes[i][2] - boxes[i][0];
        heights[i] = boxes[i][3] - boxes[i][1];
        ctr_x[i] = boxes[i][0] + 0.5*widths[i];
        ctr_y[i] = boxes[i][1] + 0.5*heights[i];
     }

     vector<float>dx(sz);
     vector<float>dy(sz);
     vector<float>dw(sz);
     vector<float>dh(sz);

     for(int i=0; i<sz; i++){
        dx[i] = rel_codes[i][0]/weights[0];
        dy[i] = rel_codes[i][1]/weights[1];
        dw[i] = min(rel_codes[i][2]/weights[2], bbox_xform_clip);
        dh[i] = min(rel_codes[i][3]/weights[3], bbox_xform_clip);
     }

     vector<float>pred_ctr_x(sz);
     vector<float>pred_ctr_y(sz);
     vector<float>pred_w(sz);
     vector<float>pred_h(sz);

     for(int i=0; i<sz; i++){
        pred_ctr_x[i] = dx[i]*widths[i]+ctr_x[i];
        pred_ctr_y[i] = dy[i]*heights[i]+ctr_y[i];
        pred_w[i] = exp(dw[i])*widths[i];
        pred_h[i] = exp(dh[i])*heights[i];
     }
     for(int i=0; i<sz; i++){
       res[i][0] = min(max(pred_ctr_x[i] - pred_w[i]*0.5, 0.0), (double)image_size);
       res[i][1] = min(max(pred_ctr_y[i] - pred_h[i]*0.5, 0.0), (double)image_size);
       res[i][2] = min(max(pred_ctr_x[i] + pred_w[i]*0.5, 0.0), (double)image_size);
       res[i][3] = min(max(pred_ctr_y[i] + pred_h[i]*0.5, 0.0), (double)image_size);
     }

     return res;
}

void find_topk_num_bigger_than_threshold(vector<float>&scores_per_level, vector<pair<float,int>>&vals_idxs,
float &threshold, int &topk_num){
    int n = scores_per_level.size();
    vector<pair<float,int>>vals_idxs;
    for(int i=0; i<n;i++){
      int val = scores_per_level[i];
      if(val>threshold){
        vals_idxs.push_back({val,i});
      }
    }

    sort(vals_idxs.begin(), vals_idxs.end(), [](pair<float, int>&a, pair<float, int>&b){
      if(a.first==b.first){
        return a.second<b.second;
      }
      return a.first>b.first;
    });

    if(vals_idxs.size()>topk_num){
      vals_idxs.erase(vals_idxs.begin()+topk_num, vals_idxs.end());
    }
}

void find_anchors_labels(vector<int>&anchor_idxs, vector<int>&labels_per_level, vector<pair<float,int>>&vals_idxs){
     for(int i=0; i<vals_idxs.size(); ++i){
          auto&[a,b] = vals_idxs[i];
          anchor_idxs[i] = b/num_classes;
          labels_per_level[i] = b%num_classes;
     }
}

template <typename T>
T& find_index(T&in, vector<int>&idxs){
    T res;
    for(auto id:idxs){
      res.push_back(in[id]);
    }
    return res;
}
void post_process(vector<float>&images, vector<float>&feat0, vector<float>&feat1, 
vector<float>&feat2, vector<float>&feat3, vector<float>&feat4, vector<float>&head00, 
vector<float>&head01, vector<float>&head02, vector<float>&head03, vector<float>&head04, 
vector<float>&head10, vector<float>&head11, vector<float>&head12, vector<float>&head13,
vector<float>&head14)
{
    /*[[k, 90000, 264],[k, 22500, 264],[k, 5625,264],[k, 1521,264],[k, 441,264]]*/
    vector<vector<float>> cls_logits(5);
    /*[[k, 90000, 4],[k, 22500, 4],[k, 5625,4],[k, 1521,4],[k, 441,4]]*/
    vector<vector<float>> bbox_regression(5);

    /*[[[90000,4],[22500,4],[5625,4],[1521,4],[441,4]],
    [[90000,4],[22500,4],[5625,4],[1521,4],[441,4]]]*/
    vector<vector<vector<vector<float>>>>anchors_for_images;
    
    load_anchors(anchors_for_images);
    int in_depth = 2376;
    cls_logits.push_back(process_tensor(head00, in_depth, 100, 100));//
    cls_logits.push_back(process_tensor(head01, in_depth, 50, 50));
    cls_logits.push_back(process_tensor(head02, in_depth, 25, 25));
    cls_logits.push_back(process_tensor(head03, in_depth, 13, 13));
    cls_logits.push_back(process_tensor(head04, in_depth, 7, 7));
    
    int in_depth = 36;
    bbox_regression.push_back(process_tensor(head00, in_depth, 100, 100));
    bbox_regression.push_back(process_tensor(head01, in_depth, 50, 50));
    bbox_regression.push_back(process_tensor(head02, in_depth, 25, 25));
    bbox_regression.push_back(process_tensor(head03, in_depth, 13, 13));
    bbox_regression.push_back(process_tensor(head04, in_depth, 7, 7));
    
    //make 5 level of one image boxes, scores, labels into one dimesnion
    vector<vector<float>>image_boxes;
    vector<float>image_scores;
    vector<int>image_labels;

    //some parameters
    float nms_thresh = 0.5;
    int topk_num = 1000;
    float score_thresh = 0.05; 
    int level_num=5;
    //final detection results
    vector<detection>res(num_images);
    vector<int>dims_scores={90000*264, 22500*264, 5625*264, 1521*264, 441*264};
    vector<int>dims_boxes={90000*4, 22500*4, 5625*4, 1521*4, 441*4};
    for(int i=0; i<num_images; i++){
      for(int j=0; j<level_num; j++){

        /*choose the value which is bigger than threshold and sort the value and 
        find the index*/
        vector<pair<float,int>>vals_idxs;
        vector<float>scores_per_level = sigmoid(cls_logits[j], i, dims_scores[j]);

        find_topk_num_bigger_than_threshold(scores_per_level, vals_idxs, score_thresh, topk_num);
        
        int sz_id = vals_idxs.size();

        //find the id belong to which anchor
        //find the id belong to which class
        vector<int>anchor_idxs(sz_id);
        vector<int>labels_per_level(sz_id);
        find_anchors_labels(anchor_idxs, labels_per_level, vals_idxs);
  
        //convert to the predicted box and clip it
        vector<vector<float>>anchors_per_level = anchors_for_images[i][j];
        vector<vector<float>>boxes_per_level(sz_id, vector<float>(4));

        boxes_per_level = box_to_picture_clip(anchor_idxs, bbox_regression[j], dims_boxes[j], i, anchors_per_level, image_size);
        
        // vector<vector<float>>IB = clip_boxes_to_image();
        
        image_boxes.insert(image_boxes.end(), boxes_per_level.begin(), boxes_per_level.end());
        image_scores.insert(image_scores.end(), scores_per_level.begin(), scores_per_level.end());
        image_labels.insert(image_labels.end(), labels_per_level.begin(), labels_per_level.end());
      }
      
      int offset = image_size+1;
      int final_temp_n = image_labels.size();
      vector<vector<float>>image_boxes_nms(final_temp_n, vector<float>(4));
      
      for(int i=0; i<final_temp_n; i++){
          int offset_temp = offset*image_labels[i];
          image_boxes_nms[i][0] = image_boxes[i][0]+offset_temp;
          image_boxes_nms[i][1] = image_boxes[i][1]+offset_temp;
          image_boxes_nms[i][2] = image_boxes[i][2]+offset_temp;
          image_boxes_nms[i][3] = image_boxes[i][3]+offset_temp;
      }

      vector<int>keep_idxs = NMS(image_boxes_nms, image_scores, nms_thresh);
      find_index(image_boxes, keep_idxs);
      detection dec = {find_index(image_boxes, keep_idxs), find_index(image_scores, keep_idxs),
      find_index(image_labels, keep_idxs)};
      res.push_back(dec);
    } 

}


int main(){
    // post_process(images);
    cout<<"sds"<<endl;
}