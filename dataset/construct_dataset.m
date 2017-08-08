clear all; close all; clc;
pover=zeros(35*10,192*192+1,10);
for u=1:4%u个字
for z=1:20%z个样图
    
    file_name=strcat('E:\Data\try\images\',num2str(u),'\',num2str(z),'.jpeg');
    img=imread(file_name);
    thresh = graythresh(img);     %自动确定二值化阈值；
    o = imbinarize(img,thresh);
    img=double(o);
    imshow(img,[]);
    [m, n]=size(img);
    
    %伽马校正
    img=sqrt(img);
    
    %下面是求边缘
    fy=[1 2 1;
        0 0 0;
        -1 -2 -1];        %定义竖直模板
    fx=-fy';             %定义水平模板
    
    
    %卷积
    Iy=imfilter(img,fy,'same');    %replicate or X
    Ix=imfilter(img,fx,'same'); %水平边缘
    G=sqrt(Ix.^2+Iy.^2);           %amplitude of gradient
    
    
    %求边缘强度与斜率
    Ied=sqrt(Ix.^2+Iy.^2);              %边缘强度
    Iphase=Iy./Ix;              %边缘斜率，有些为inf,-inf,nan，其中nan需要再处理一下
    
    %创建八个空白图
    p(:,:,1)=img;
    p(:,:,2)=G;
    for l=3:10
        p(:,:,l)=zeros(n);
    end
    angle=zeros(n);
    
    
    %判定及填充
    step=1;                %step滑动步长
    
    for i=1:step:n          %如果处理的m/step不是整数，最好是i=1:step:m-step
        
        for j=1:step:n      %注释同上
            
            if isnan(Iphase(i,j))==1  %0/0会得到nan，如果像素是nan，重设为0
                Iphase(i,j)=0;
            end
            ang=atan(Iphase(i,j));    %atan求的是[-90 90]度之间
            ang=mod(ang*180/pi,360);    %全部变正，-90变270
            if Ix(i,j)<0              %根据x方向确定真正的角度
                if ang<90               %如果是第一象限
                    ang=ang+180;        %移到第三象限
                end
                if ang>270              %如果是第四象限
                    ang=ang-180;        %移到第二象限
                end
            end
            angle(i,j)=ang;
            %分解向量大小
            a1= abs(Ix(i,j)-Iy(i,j));
            a2=min(abs(Ix(i,j)),abs(Iy(i,j)));
            
            
            %判别方向并填充
            if 0<=ang && ang<=45 %在第一个方向中
                p(i,j,3)=Ix(i,j);
                % picture_0(i,j)=G(i,j);
                p(i,j,4)=a1;
                
            elseif 45< ang && ang <=90
                p(i,j,4)=a1;
                p(i,j,5)=a2;
                % picture_45(i,j)=G(i,j);
                
            elseif 90<ang && ang<=135
                p(i,j,5)=a2;
                p(i,j,6)=a1;
                %  picture_90(i,j)=G(i,j);
                
            elseif 135<ang && ang<=180
                %  picture_135(i,j)=G(i,j);
                p(i,j,6)=a1;
                p(i,j,7)=a2;
                
            elseif 180<ang && ang<=225
                %  picture_180(i,j)=G(i,j);
                p(i,j,7)=a2;
                p(i,j,8)=a1;
                
            elseif 225<ang && ang<=270
                %   picture_225(i,j)=G(i,j);
                p(i,j,8)=a1;
                p(i,j,9)= a2;
                
            elseif 270<ang && ang<=315
                %   picture_270(i,j)=G(i,j);
                p(i,j,9)= a2;
                p(i,j,10)=a1;
                
            elseif 315< ang && ang<=360
                % picture_315(i,j)=G(i,j);
                p(i,j,10)=a1;
                p(i,j,3)=a2;
            end
            
        end
        
    end%得到p（n，n，10）即十个图
    
    %清晰图像
    %for l=2:10
    %     p(:,:,l)=255*p(:,:,l)./max(p(:,:,l));
    % end
    
    %叠二维标label
    picture=zeros(n*n+1,10);
    
    for k=1:10
        picture(1:n*n,k)=reshape(p(:,:,k),1,n*n);
        picture(n*n+1,k)=u-1;
    end
    %叠三维
    x=(u-1)*35+z;%行数
    pover(x,:,:)=picture;
end
end
%save  testdata pover