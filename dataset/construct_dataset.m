clear all; close all; clc;
pover=zeros(35*10,192*192+1,10);
for u=1:4%u����
for z=1:20%z����ͼ
    
    file_name=strcat('E:\Data\try\images\',num2str(u),'\',num2str(z),'.jpeg');
    img=imread(file_name);
    thresh = graythresh(img);     %�Զ�ȷ����ֵ����ֵ��
    o = imbinarize(img,thresh);
    img=double(o);
    imshow(img,[]);
    [m, n]=size(img);
    
    %٤��У��
    img=sqrt(img);
    
    %���������Ե
    fy=[1 2 1;
        0 0 0;
        -1 -2 -1];        %������ֱģ��
    fx=-fy';             %����ˮƽģ��
    
    
    %���
    Iy=imfilter(img,fy,'same');    %replicate or X
    Ix=imfilter(img,fx,'same'); %ˮƽ��Ե
    G=sqrt(Ix.^2+Iy.^2);           %amplitude of gradient
    
    
    %���Եǿ����б��
    Ied=sqrt(Ix.^2+Iy.^2);              %��Եǿ��
    Iphase=Iy./Ix;              %��Եб�ʣ���ЩΪinf,-inf,nan������nan��Ҫ�ٴ���һ��
    
    %�����˸��հ�ͼ
    p(:,:,1)=img;
    p(:,:,2)=G;
    for l=3:10
        p(:,:,l)=zeros(n);
    end
    angle=zeros(n);
    
    
    %�ж������
    step=1;                %step��������
    
    for i=1:step:n          %��������m/step���������������i=1:step:m-step
        
        for j=1:step:n      %ע��ͬ��
            
            if isnan(Iphase(i,j))==1  %0/0��õ�nan�����������nan������Ϊ0
                Iphase(i,j)=0;
            end
            ang=atan(Iphase(i,j));    %atan�����[-90 90]��֮��
            ang=mod(ang*180/pi,360);    %ȫ��������-90��270
            if Ix(i,j)<0              %����x����ȷ�������ĽǶ�
                if ang<90               %����ǵ�һ����
                    ang=ang+180;        %�Ƶ���������
                end
                if ang>270              %����ǵ�������
                    ang=ang-180;        %�Ƶ��ڶ�����
                end
            end
            angle(i,j)=ang;
            %�ֽ�������С
            a1= abs(Ix(i,j)-Iy(i,j));
            a2=min(abs(Ix(i,j)),abs(Iy(i,j)));
            
            
            %�б������
            if 0<=ang && ang<=45 %�ڵ�һ��������
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
        
    end%�õ�p��n��n��10����ʮ��ͼ
    
    %����ͼ��
    %for l=2:10
    %     p(:,:,l)=255*p(:,:,l)./max(p(:,:,l));
    % end
    
    %����ά��label
    picture=zeros(n*n+1,10);
    
    for k=1:10
        picture(1:n*n,k)=reshape(p(:,:,k),1,n*n);
        picture(n*n+1,k)=u-1;
    end
    %����ά
    x=(u-1)*35+z;%����
    pover(x,:,:)=picture;
end
end
%save  testdata pover