%% PR?CTICA 7, Decision Trees for the Brain database
clear
%close all
clc 
load PRBB_Brain
i_dib=0;                     % 1 Dibuja las im?genes
i_valida_split=0;            % 1 Valida n?mero m?ximo de splits in trees
rng('shuffle')

%% Se dibujan opcionalmente las im?genes
N_images=8;
Ndim=256;
N_datos=Ndim*Ndim;
Brain_8=zeros(N_datos,N_images);
for i1=1:N_images
    Vaux=Brain(:,i1);
    Aux=reshape(Vaux,Ndim,Ndim);
    if i_dib==1
        figure
        imagesc(Aux)
        axis image
        colorbar
    end
    Brain_8(:,i1)=Aux(:);
end
clear Vaux Aux N_images i1 Brain
%%  Se etiquetan los vectores (p?xeles) utilizando las probabilidades de las clases (images 6,7,8)
N_feat=5;
Brain_5=Brain_8(:,1:N_feat);

% Detecci?n de p?xeles de fondo 'clase 4'
[class,ind]=max(Brain_8(:,6:8),[],2);
Index_clase0=find(class==0);
Brain_Etiq(1:length(Index_clase0),1:N_feat)=Brain_5(Index_clase0,1:N_feat);
Labels(1:length(Index_clase0))=4*ones(length(Index_clase0),1);

%Detecci?n del resto de p?xeles etiquetados
Pr_min=0.9;
Index_Labels=find(class>=Pr_min);
Brain_Etiq = [Brain_Etiq ; Brain_5(Index_Labels,1:N_feat)];
Labels(length(Index_clase0)+1: length(Index_clase0)+length(Index_Labels))=ind(Index_Labels);
%
if i_dib==1
    %Representaci?n de BD Etiquetada.
    Labeled_Image=zeros(N_feat,1);
    Labeled_Image(Index_clase0)=4;
    Labeled_Image(Index_Labels)=ind(Index_Labels);
    Aux=reshape(Labeled_Image,Ndim,Ndim);
    figure('name','Labeled Image')
    imagesc(Aux)
    axis image
    colorbar
    clear Labeled_Image Aux class ind
end
clear Brain_8 class ind i_dib

%% BD
% Eliminar excedentes de class=4 (Background)
N_classes=4;
N_size=zeros(1,N_classes);
Brain_Etiq2=[];
Labels2=[];
for i_class=1:N_classes-1
    N_size(i_class)=length(find(Labels==i_class));
    Brain_Etiq2=[Brain_Etiq2;Brain_Etiq(find(Labels==i_class),:)];
    Labels2=[Labels2 i_class*ones(1,N_size(i_class))];
end
N_c4=round(mean(N_size(1:N_classes-1))); %Reduccion de numero elementos clase fondo
Vaux=find(Labels==N_classes);
Brain_Etiq2=[Brain_Etiq2;Brain_Etiq(Vaux(1:N_c4),:)];
Labels2=[Labels2 N_classes*ones(1,N_c4)]; %Nuevo vector de etiquetas
clear Vaux N_c4 N_size indexperm i_class
Brain_Etiq=Brain_Etiq2;
Labels=Labels2;
clear Labels2 Brain_Etiq2 Index_clase0 Index_Labels %Index_NO_label

%Aleatorizaci?n orden de los vectores
Permutation=randperm(length(Labels));
Brain_Etiq=Brain_Etiq(Permutation,:);
Labels=Labels(Permutation);

%% Generaci?n ?ndices de BD Train, BD Val, BD Test
P_train=0.6;
P_val=0.2;
P_test=1-P_train-P_val;
Index_train=[];
Index_val=[];
Index_test=[];
Labels=Labels';
for i_class=1:N_classes
    index=find(Labels==i_class);
    N_i_class=length(index);
    [I_train,I_val,I_test] = dividerand(N_i_class,P_train,P_val,P_test);
    Index_train=[Index_train;index(I_train)];
    Index_val=[Index_val;index(I_val)];
    Index_test=[Index_test;index(I_test)];
end
% Mixing of vectors not to have all belonging to a class together
Permutation=randperm(length(Index_train));
Index_train=Index_train(Permutation);
Permutation=randperm(length(Index_val));
Index_val=Index_val(Permutation);
Permutation=randperm(length(Index_test));
Index_test=Index_test(Permutation);
clear Permutation i_class index N_i_class I_train I_val I_test

% Generaci?n BD Train, BD CV, BD Test
X_train=Brain_Etiq(Index_train,:);
Labels_train=Labels(Index_train);
X_val=Brain_Etiq(Index_val,:);
Labels_val=Labels(Index_val);
X_test=Brain_Etiq(Index_test,:);
Labels_test=Labels(Index_test);


%% TREE CLASSIFIER
 
% Tree classifier design
tree = fitctree(X_train,Labels_train);
view(tree,'mode','graph');
view(tree)
    
% Measure Train error
outputs = predict(tree,X_train);
Tree_Pe_train=sum(Labels_train ~= outputs)/length(Labels_train);
fprintf('\n------- TREE CLASSIFIER ------------------\n')   
fprintf(1,' error Tree train = %g   \n', Tree_Pe_train)  
CM_Train=confusionmat(Labels_train,outputs)
% Measure Val error
outputs = predict(tree,X_val);
Tree_Pe_val=sum(Labels_val ~= outputs)/length(Labels_val);
fprintf('\n-------------------------\n')   
fprintf(1,' error Tree val = %g   \n', Tree_Pe_val)  
CM_Val=confusionmat(Labels_val,outputs)
% Measure Test error
outputs = predict(tree,X_test);
Tree_Pe_test=sum(Labels_test ~= outputs)/length(Labels_test);
fprintf('\n-------------------------\n')   
fprintf(1,' error Tree test = %g   \n', Tree_Pe_test)
CM_Test=confusionmat(Labels_test,outputs)

%% TREE IMAGE CLASSIFICATION
outputs = predict(tree,Brain_5);
Aux=reshape(outputs,Ndim,Ndim);
figure('name','Tree Classified Image')
imagesc(Aux)
axis image
colorbar
clear Aux outputs

%% Validacion de n?mero M?XIMO DE SPLITS - Tree
if i_valida_split==1;
    
    % TO DO, FOR MIN LEAVES AND TOP-DOWN CRITERIA
    % for ParameterValues = 
    %   Train a tree with the train BD and the train targets
    %   Measure Train, Val and Test classification errors
    %   Keep or save the tree classifier associated to the minimum val
    %   error
    % end for
    % Plot train, val and test errors for each value of the parameter
    
    % END TO DO
end
clear i_valida_split
clear Index_train Index_val Index_test