#include<vector>
#include<string>
#include<fstream>
#include<iostream>
#include <algorithm>
#include<cstdint>

using namespace std;

class Matrix{
    private:
        vector<float> data;

    public:
        int rows;
        int cols;

        Matrix(int r, int c){
            rows = r;
            cols = c;
            data.resize(rows * cols);
        }

        float get(int r, int c) const{
            return data[r * cols + c];
        }

        void set(int r, int c, float val){
            data[r * cols + c] = val;
        }
        
        bool load_from_file(const string &filename){
            ifstream file(filename, ios::binary);
                
            if(!file.is_open()){
                cerr << "Error: Could not open " << filename << endl;
                return false;
            }
        
            int total_elements = rows * cols;
            int total_bytes = total_elements * sizeof(float);
        
            // The Magic Dump: Read the raw bytes directly into the vector's memory space.
            // .data() gives us the raw memory address of our vector.
            file.read(reinterpret_cast<char*>(data.data()), total_bytes);
        
            file.close();
            return true;
        }

        void add_bias(const Matrix &bias){
            if(this->cols != bias.cols){
                cerr << "Error: Bias columns do not match Matrix columns!" << endl;
                exit(1);
            }

            for(int r = 0; r < rows; ++r){
                for(int c = 0; c < cols; ++c){
                    float new_val = this->get(r,c) + bias.get(0,c);
                    this->set(r,c, new_val);
                }
            }
        }

        void apply_relu(){
            for(int i = 0; i < rows * cols; ++i){
                data[i] = max(0.0f, data[i]);
            }
        }

};

Matrix multiply(const Matrix &A, const Matrix &B){
    if(A.cols != B.rows){
        cerr << "Error: Matrix dimensions don't match for multiplication!" << endl;
        exit(1);
    }

    Matrix C(A.rows, B.cols);

    for(int i = 0; i < A.rows; ++i){
        for(int j = 0; j < B.cols; ++j){
            float sum = 0.0;
            for(int k = 0; k < A.cols; ++k){
                sum += (A.get(i,k)*B.get(k,j));
            }
            C.set(i, j , sum);
        }
    }
    return C;
}

uint32_t swap_endian(uint32_t val){
    return ((val << 24) & 0xff000000) |
           ((val << 8) & 0x00ff0000) |
           ((val >> 8) & 0x0000ff00) |
           ((val >> 24) & 0x0000ff);
}

Matrix load_mnist_image(const string &filepath, int image_index){
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        cerr << "Failed to open MNIST dataset!" << endl;
        exit(1);
    }
    
    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);
                            
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    int image_size = rows * cols;

    file.seekg(16 + (image_index * image_size), ios::beg);

    vector<uint8_t> raw_pixels(image_size);
    file.read(reinterpret_cast<char*>(raw_pixels.data()), image_size);

    Matrix x(1, image_size);
    for(int i = 0; i < image_size; ++i){
        x.set(0, i, raw_pixels[i] / 255.0f);
    }
    
    return x;
}

Matrix load_custom_image(const string &filepath){
    ifstream file(filepath);
    if(!file.is_open()){
        cerr << "Failed to open custom image file!" << endl;
        exit(1);
    }
    Matrix x(1, 784);
    float val;
    for(int i = 0; i < 784; ++i){
        file >> val;
        x.set(0, i, val);
    }
    return x;
}

int load_mnist_label(const string& filepath, int label_index){
    ifstream file(filepath, ios::binary);
    if(!file.is_open()){
        cerr << "Failed to open label file!" << endl;
        exit(1);
    }

    file.seekg(8 + label_index, ios::beg);

    uint8_t label;
    file.read(reinterpret_cast<char*> (&label), 1);

    return static_cast<int> (label);
}

int main(int argc, char* argv[]){
    cout << "--- Initializing C++ Inference Engine ---" << endl;

    //weights
    Matrix W1(784, 64);
    if (!W1.load_from_file("weights_archive/w1.bin")) exit(1);
    W1.load_from_file("weights_archive/w1.bin");
    Matrix b1(1, 64);
    if (!b1.load_from_file("weights_archive/b1.bin")) exit(1);
    W1.load_from_file("weights_archive/w1.bin");
    Matrix W2(64, 10);
    if (!W2.load_from_file("weights_archive/w2.bin")) exit(1);
    W1.load_from_file("weights_archive/w1.bin");
    Matrix b2(1, 10);
    if (!b2.load_from_file("weights_archive/b2.bin")) exit(1);
    W1.load_from_file("weights_archive/w1.bin");

    cout << "Weights loaded successfully." << endl;

    if(argc == 2){
        string filepath = argv[1];
        Matrix x = load_custom_image(filepath);

        std::cout << "\n--- What the AI is actually seeing ---" << std::endl;
        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                float pixel = x.get(0, r * 28 + c); 
                if (pixel > 0.5f) std::cout << "██";       
                else if (pixel > 0.1f) std::cout << "░░";  
                else std::cout << "  ";                    
            }
            std::cout << std::endl;
        }
        std::cout << "--------------------------------------\n" << std::endl;

        Matrix y1 = multiply(x, W1);
        y1.add_bias(b1);
        y1.apply_relu();

        Matrix y2 = multiply(y1, W2);
        y2.add_bias(b2);

        int best_digit = 0;
        float best_score = y2.get(0,0);
        for(int j = 0; j < 10; ++j){
            float score = y2.get(0, j);
            if(score > best_score){
                best_score = score;
                best_digit = j;
            }
        }
        cout << "PREDICTION: " << best_digit << endl;
        return 0;
    }

//---------------------------------------------------------------------------------------------------------

    //BENCHMARK TEST

    // int correct_guesses = 0;
    // int total_images = 1000;

    // for(int i = 0; i < total_images; ++i){
    //     int correct = load_mnist_label("t10k-labels-idx1-ubyte", i);
    //     Matrix x = load_mnist_image("t10k-images-idx3-ubyte", i);    

    //     Matrix y1 = multiply(x, W1);
    //     y1.add_bias(b1);
    //     y1.apply_relu();

    //     Matrix y2 = multiply(y1, W2);
    //     y2.add_bias(b2);

    //     int best_digit = 0;
    //     float best_score = y2.get(0,0);
    //     for(int j = 0; j < 10; ++j){
    //         float score = y2.get(0, j);
    //         if(score > best_score){
    //             best_score = score;
    //             best_digit = j;
    //         }
    //     }

    //     if(best_digit == correct) correct_guesses++;

    //     if(i % 1000 != 0){
    //         cout << "Testing.. Image " << i << " / " << total_images << endl;
    //     }
    // }

    // float accuracy = (correct_guesses / (float)total_images) * 100.0f;

    // cout << "\n===================================" << endl;
    // cout << "       BENCHMARK COMPLETE          " << endl;
    // cout << "===================================" << endl;
    // cout << "Total Correct: " << correct_guesses << " / " << total_images << endl;
    // cout << "Final Accuracy: " << accuracy << "%" << endl;

//----------------------------------------------------------------------------------------------------------

    //SINGLE IMAGE FORWARD PASS

    // Matrix x = load_mnist_image("train-images-idx3-ubyte", 10);
    // cout << "Image 0 loaded and normalized." << endl;

    // std::cout << "\n--- Visualizing What the AI Sees ---" << std::endl;
    // for (int r = 0; r < 28; ++r) {
    //     for (int c = 0; c < 28; ++c) {
    //         // Grab the pixel from our flat 1x784 matrix
    //         float pixel = x.get(0, r * 28 + c); 
            
    //         // Print different characters based on pixel brightness
    //         if (pixel > 0.5f) std::cout << "██";       // Solid white
    //         else if (pixel > 0.1f) std::cout << "░░";  // Gray/Edge
    //         else std::cout << "  ";                    // Black background
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "------------------------------------\n" << std::endl;

    // //forward pass
    // Matrix y1 = multiply(x, W1);
    // y1.add_bias(b1);
    // y1.apply_relu();

    // Matrix y2 = multiply(y1, W2);
    // y2.add_bias(b2);

    // cout << "\n--- Final Network Scores (Logits) ---" << endl;
    // int best_digit = 0;
    // float best_score = y2.get(0, 0);

    // for(int i = 0; i < 10; ++i){
    //     float score = y2.get(0, i);
    //     cout << "Digit " << i << ": " << score << endl;

    //     if(score > best_score){
    //         best_score = score;
    //         best_digit = i;
    //     }
    // }

    // cout << "\n>>> PREDICTION: The network thinks this is a " << best_digit << " <<<\n" << endl;
    return 0;
}