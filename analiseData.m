data = csvread('PhishingData.txt')

%sum(data(:,1)==1;
%sum(data(:,1)==0;
%sum(data(:,1)==-1;
features=[];
dataSize = size(M);
%binranges=[-1,0,1]
numberOfFeatures = dataSize(1,2)-1;

for feature = 1:numberOfFeatures
    features=[features ; [sum(data(:,feature)==-1) sum(data(:,feature)==0) sum(data(:,feature)==1)]]
   % bincounts = histcounts(M(:,feature))
   figure() 
   hist(M(:,feature))    
end

rel =[  features(:,1)./ features(:,2) features(:,2)./features(:,3) ]
relF = rel(:,1)./rel(:,2);
tempR1=relF<2;
tempR2=relF>0.4;
bestFeatures=tempR1.*tempR2;
