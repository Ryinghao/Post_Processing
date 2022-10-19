// vector<vector<int>>sizes = {{32, 40, 50}, {64, 80, 101}, {128, 161, 203}, {256, 322, 406}, {512, 645, 812}};
    // vector<float>aspect_ratios={05, 10, 20};
    // vector<int>scales = {32, 40, 50};
    // vector<float>h_ratio;
    // vector<float>w_ratios;
// h_ratioresize(N);
        // w_ratiosresize(N);
        // base_anchorsresize(N*N, vector<int>(4));
        // for(int i=0; i<N; i++){
        //     h_ratio{i} = float(sqrt(scales{i}));
        //     w_ratios{i} = 1/h_ratio{i};
        // }
        // for(int i=0; i<N; i++){
        //     for(int j=0; j<N; j++){
        //         int ws = h_ratio{i}*scales{j}/2;
        //         int hs = w_ratios{i}*scales{j}/2;
        //         base_anchors{i}={-ws, -hs, ws ,hs};
        //     }
        // }

// void post_process(float(*images)[N][3][800][800],
//         float(*feat0)[N][256][100][100], float(*feat1)[N][256][50][50], float(*feat2)[N][256][25][25], 
//         float(*feat3)[N][256][13][13], float(*feat4)[N][256][7][7],
//         float(*head00)[N][2376][100][100], float(*head01)[N][2376][50][50], float(*head02)[N][2376][25][25],
//         float(*head03)[N][2376][13][13], float(*head04)[N][2376][7][7], 
//         float(*head10)[N][36][100][100], float(*head11)[N][36][50][50], float(*head12)[N][36][25][25], 
//         float(*head13)[N][36][13][13], float(*head14)[N][36][7][7])
//         {
            
//         }

// void post_process(int(*images)[N][3][800][800]){
//     int blue[2][3]  = {{0,-1,1}, {0,-1,1}};
//     // cout<<blue[1][2]<<endl;
//     unordered_map<string,int(*)[2][3]> colours; 
//     unordered_map<int,int(*)[3]> colours2; 
//     // colours.insert(std::pair<int,int(*)[2][3]>(1,&blue));
//     // colours2.insert(std::pair<int, int(*)[3]>(1,&blue2));
//     colours["cls_logits"] = &blue;
//     // cout<<&colours2<<endl;
//     // cout<<(*colours2[1])[1]<<endl;
//     cout<<(*colours["cls_logits"])[0][2]<<endl;
//     // mp["cls_logits"] = head_outputs_list[0];

// }
int array[5][3][3];
int blue2[3]  = {-1,2,-1};
float images[len_image][3][800][800];
float features[len_image][3][800][800];
float head_outputs_list[len_image][3][800][800];