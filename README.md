# DimensionalityReduction_alo_codes

### **网上关于各种降维算法的资料参差不齐，同时大部分不提供源代码；在此通过借鉴资料实现了一些经典降维算法的Demo(python)，同时也给出了参考资料的链接。**

### 现已实现算法:

1. PCA/KPCA

- https://blog.csdn.net/u013719780/article/details/78352262
- https://blog.csdn.net/weixin_40604987/article/details/79632888
![PCA](codes/PCA/PCA.png)
![KPCA](codes/PCA/KPCA.png)
2. LDA
- https://blog.csdn.net/ChenVast/article/details/79227945
- https://www.cnblogs.com/pinard/p/6244265.html
![LDA](codes/LDA/LDA.png)

3. MDS
- https://blog.csdn.net/zhangweiguo_717/article/details/69663452?locationNum=10&fps=1

![MDS](codes/MDS/MDS_1.png)
![Tensor-MDS](codes/MDS/MDS_2.png)
4. ISOMAP
- https://blog.csdn.net/zhangweiguo_717/article/details/69802312
- http://www-clmc.usc.edu/publications/T/tenenbaum-Science2000.pdf
![ISOMAP](codes/ISOMAP/Isomap.png)
5. LLE
- https://blog.csdn.net/scott198510/article/details/76099630
- https://www.cnblogs.com/pinard/p/6266408.html?utm_source=itdadao&utm_medium=referral
![LLE](codes/LLE/LLE.png)
6. TSNE
- http://bindog.github.io/blog/2018/07/31/t-sne-tips/
![TSNE](codes/T-SNE/T-SNE.png)
7. AutoEncoder
![AutoEncoder](codes/AutoEncoder/AutoEncoder.png)
8. FastICA
- https://blog.csdn.net/lizhe_dashuju/article/details/50263339

9. SVD
- https://blog.csdn.net/m0_37870649/article/details/80547167
- https://www.cnblogs.com/pinard/p/6251584.html

10. LE
![LE](codes/LE/LE.png)
实现了之后发现这个LLE算法鲁棒性太太差了，完全没法用。对参数和数据非常非常非常敏感!!!
- https://blog.csdn.net/hustlx/article/details/50850342#
- https://blog.csdn.net/jwh_bupt/article/details/8945083