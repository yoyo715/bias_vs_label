

with open('/local_d/RESEARCH/bias_vs_eff/bias_vs_labelefficiency/TEST.txt') as f:
    content = f.readlines()
    
i = 0
for line in content:
    if "Fetched" in line:
        i +=  1
        
print("Number fetched = ", i)
