/*
Big-EndianMethod: Stores the highest value byte at the lowest address (left-to-right).
Example: The 4-byte hex number 0x12345678 is stored as 12 34 56 78.
Usage: Often used in network protocols (TCP/IP) and high-level languages.
Pros: Intuitive for humans, as it matches how we write numbers. 

Little-EndianMethod: Stores the lowest value byte at the lowest address (right-to-left).
Example: The 4-byte hex number 0x12345678 is stored as 78 56 34 12.
Usage: Used by most modern processors, including Intel and ARM processors.
Pros: Facilitates easier arithmetic and pointer manipulation on the CPU.
*/

#include<iostream>
#include<fstream>
#include<vector>
#include<cstdint>

using namespace std;

uint32_t swap_endian(uint32_t val){
    return ((val << 24) & 0xff000000) |
           ((val << 8) & 0x00ff0000) |
           ((val >> 8) & 0x0000ff00) |    
           ((val >> 24) & 0xff0000ff);
}

int main(){
// Open the file in binary mode
    ifstream file("train-images-idx3-ubyte", ios::binary);
    
    if (!file.is_open()) {
        cerr << "Error: Could not open the MNIST file. Check the path." << std::endl;
        return 1;
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    // Read the 16-byte header (4 integers, 4 bytes each)
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    // Swap the endianness for our Intel architecture
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    cout << "Magic Number: " << magic_number <<endl;
    cout << "Total Images: " << num_images << endl;
    cout << "Dimensions: " << num_rows << "x" << num_cols << "\n\n";

    // Read the first image (28 * 28 = 784 bytes)
    int image_size = num_rows * num_cols;
    vector<uint8_t> first_image(image_size);
    file.read(reinterpret_cast<char*>(first_image.data()), image_size);

    // Print the image to the terminal using ASCII characters
    cout << "--- First MNIST Image ---" << "\n\n";
    for (int r = 0; r < num_rows; ++r) {
        for (int c = 0; c < num_cols; ++c) {
            uint8_t pixel = first_image[r * num_cols + c];
            
            // Map pixel intensity to ASCII characters
            if (pixel > 200) cout << "@@";
            else if (pixel > 128) cout << "##";
            else if (pixel > 64) cout << "..";
            else cout << "  "; // empty space for black pixels
        }
        cout << "\n";
    }

    file.close();
    return 0;
}