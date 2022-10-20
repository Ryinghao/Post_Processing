#include<iostream>
#include<unordered_map>
#include<map>
#include<vector>
#include<list>
#include<algorithm>
using namespace std;

struct Box{
    int id;
    vector<float>image_box_nms;
    float score;
    double area;
    Box(int id, vector<float>&image_box_nms, float score){
        id = id;
        image_box_nms = image_box_nms;
        score = score;
        area = (image_box_nms[1]-image_box_nms[3])*(image_box_nms[2]-image_box_nms[0]);
    }
};

float IOU(Box&best_box, Box&other_box){
    int x1max = max(best_box.image_box_nms[0], other_box.image_box_nms[0]);      // the max value of two window leftup corner'x 
    int x2min = min(best_box.image_box_nms[2], other_box.image_box_nms[2]);     // the min value of two window rightdown corner'x 
    int y1min = min(best_box.image_box_nms[1], other_box.image_box_nms[1]);      // the min value of two window leftup corner'y 
    int y2max = max(best_box.image_box_nms[3], other_box.image_box_nms[3]);     // the max value of two window rightdown corner'y 
    int overlapWidth = x2min - x1max;            // 计算两矩形重叠的宽度 
    int overlapHeight = y1min - y2max;
    double intersect_area;
    double union_area;
    if(overlapWidth > 0 && overlapHeight > 0){
        intersect_area = overlapWidth*overlapHeight;
        union_area = best_box.area + other_box.area-intersect_area;
    }
    return intersect_area/union_area;
}

vector<int> NMS(vector<vector<float>>&image_boxes_nms, vector<float>image_scores, float nms_threshold){
    
    int sz = image_scores.size();
    vector<Box>Image_Boxes(sz);

    for(int i=0; i<sz; i++){
        Image_Boxes[i]=Box(i, image_boxes_nms[i], image_scores[i]);
    }

    sort(Image_Boxes.begin(), Image_Boxes.end(), [](Box a, Box b){
        return a.score>=b.score;
    });
    
    vector<int>id_res;

    while(Image_Boxes.size()){
        Box best_box = Image_Boxes.front();
        Image_Boxes.erase(Image_Boxes.end());
        id_res.push_back(best_box.id);
        for (auto iter=Image_Boxes.begin(); iter!=Image_Boxes.end();) {
            if (IOU(best_box, *iter) > nms_threshold) {
                iter = Image_Boxes.erase(iter);
            }else{
                iter++;
            }
        }
    }
    return id_res;
}