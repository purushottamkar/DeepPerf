with open('src-test.txt','w') as s:
  with open('targ-test.txt','w') as t:    
    for line in open('test.txt'):
      l = line.strip().split('\t')
      s.write(l[-1]+'\n')
      t.write(l[-2]+'\n')
