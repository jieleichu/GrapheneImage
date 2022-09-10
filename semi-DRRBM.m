% This program trains semi-DDRBM.

% maxepoch  -- maximum number of epochs
% numhid    -- number of hidden units 
% batchdata -- the data that is divided into batches (numcases numdims numbatches)
% restart   -- set to 1 if learning starts from beginning 
%ISet --inside cluster set
%OSet --outside cluster set
function [ vis hid errorS] = rbm_Graphene_binary(batchdata,numdims,numhid,ISet,OSet)
epsilonw      = 0.1;   % Learning rate for weights 
epsilonvb     = 0.1;   % Learning rate for biases of visible units 
epsilonhb     = 0.1;   % Learning rate for biases of hidden units 
weightcost  = 0.2;   
initialmomentum  = 0.5;
finalmomentum    = 0.9;
maxepoch=10;
error_return=zeros(1,maxepoch);
mu=0.8;
[numcases numdims ]=size(batchdata);
  epoch=1;
 numbatches=1;
% Initializing symmetric weights and biases. 
  vishid     = 0.1*randn(numdims, numhid);
  hidbiases  = zeros(1,numhid);
  visbiases  = zeros(1,numdims);

  poshidprobs = zeros(numcases,numhid);
  neghidprobs = zeros(numcases,numhid);
  posprods    = zeros(numdims,numhid);
  negprods    = zeros(numdims,numhid);
  vishidinc  = zeros(numdims,numhid);
  hidbiasinc = zeros(1,numhid);
  visbiasinc = zeros(1,numdims);
  
  batchposhidprobs=zeros(numcases,numhid,numbatches);

   detaW = 0*randn(numdims, numhid);
   deltaJB=0*randn(1,numhid);
   detaJrecon = 0*randn(numdims, numhid);
for epoch = epoch:maxepoch,
 fprintf(1,'epoch %d\r',epoch); 
 errsum=0;
 for batch = 1:numbatches,
 fprintf(1,'epoch %d batch %d\r',epoch,batch); 
  data = batchdata(:,:,batch);%%v_data
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));%%%h_data    
  batchposhidprobs(:,:,batch)=poshidprobs;
  posprods    = data' * poshidprobs;
  poshidact   = sum(poshidprobs);
  posvisact = sum(data);

  poshidstates = poshidprobs > rand(numcases,numhid);
  negdata = 1./(1 + exp(-poshidstates*vishid' - repmat(visbiases,numcases,1)));%%%v_recon      
  neghidprobs = 1./(1 + exp(-negdata*vishid - repmat(hidbiases,numcases,1))); %%%h_recon   
  negprods  = negdata'*neghidprobs;
  neghidact = sum(neghidprobs);
  negvisact = sum(negdata); 
  err= sum(sum( (data-negdata).^2 ));
  errsum = err + errsum;

   if epoch>5,
     momentum=finalmomentum;
   else
     momentum=initialmomentum;
   end;
   HPHQ=0;
   HPHQ_r=0;
   for loop=1:size(ISet)
       VS(loop,:)=data(ISet(loop,1),:);
       VT(loop,:)=data(ISet(loop,2),:);
       
       HS(loop,:)=poshidprobs(ISet(loop,1),:);
       HT(loop,:)=poshidprobs(ISet(loop,2),:);
       HPHQ=HPHQ+(HS(loop,:)-HT(loop,:))*(HS(loop,:)-HT(loop,:))'
       %%%%reconstruct data
       VS_r(loop,:)=negdata(ISet(loop,1),:);
       VT_r(loop,:)=negdata(ISet(loop,2),:);
       
       HS_r(loop,:)=neghidprobs(ISet(loop,1),:);
       HT_r(loop,:)=neghidprobs(ISet(loop,2),:); 
       HPHQ_r=HPHQ_r+(HS_r(loop,:)-HT_r(loop,:))*(HS_r(loop,:)-HT_r(loop,:))'
   end
 

    HDHE=0;
   HDHE_r=0;
   for loop=1:size(OSet)
       VD(loop,:)=data(OSet(loop,1),:);
       VE(loop,:)=data(OSet(loop,2),:);
       
       HD(loop,:)=poshidprobs(OSet(loop,1),:);
       HE(loop,:)=poshidprobs(OSet(loop,2),:);
       HDHE=HDHE+(HD(loop,:)-HE(loop,:))*(HD(loop,:)-HE(loop,:))'
       %%%%reconstruct data
       VD_r(loop,:)=negdata(OSet(loop,1),:);
       VE_r(loop,:)=negdata(OSet(loop,2),:);
       
       HD_r(loop,:)=neghidprobs(OSet(loop,1),:);
       HE_r(loop,:)=neghidprobs(OSet(loop,2),:); 
       HDHE_r=HDHE_r+(HD_r(loop,:)-HE_r(loop,:))*(HD_r(loop,:)-HE_r(loop,:))'
   end
 
   if HPHQ==0 || HPHQ_r==0 || HDHE==0 || HDHE_r==0 
       return;
   end
                                                                                                     
   %%%%%%%%%%%%%%%%%%%%%% compute detaW 
   for pp=1:size(HS)
        detaW=detaW+2*(VS(pp,:)'*sum(diag(HS(pp,:)-HT(pp,:))*diag(HS(pp,:))*diag(1-HS(pp,:)))-...
                                                        VT(pp,:)'*sum(diag(HS(pp,:)-HT(pp,:))*diag(HT(pp,:))*diag(1-HT(pp,:))))/(size(HS,1)*HPHQ);
   
        detaW=detaW+2*(VS_r(pp,:)'*sum(diag(HS_r(pp,:)-HT_r(pp,:))*diag(HS_r(pp,:))*diag(1-HS_r(pp,:)))-...
                                                        VT_r(pp,:)'*sum(diag(HS_r(pp,:)-HT_r(pp,:))*diag(HT_r(pp,:))*diag(1-HT_r(pp,:))))/(size(HT_r,1)*HPHQ_r);

   end

   for pp=1:size(HD)
        detaW=detaW-2*(VD(pp,:)'*sum(diag(HD(pp,:)-HE(pp,:))*diag(HD(pp,:))*diag(1-HD(pp,:)))-...
                                                        VE(pp,:)'*sum(diag(HD(pp,:)-HE(pp,:))*diag(HE(pp,:))*diag(1-HE(pp,:))))/(size(HD,1)*HDHE);
    
        detaW=detaW-2*(VD_r(pp,:)'*sum(diag(HD_r(pp,:)-HE_r(pp,:))*diag(HD_r(pp,:))*diag(1-HD_r(pp,:)))-...
                                                        VE_r(pp,:)'*sum(diag(HD_r(pp,:)-HE_r(pp,:))*diag(HE_r(pp,:))*diag(1-HE_r(pp,:))))/(size(HE_r,1)*HDHE_r);
                                                 
   end
 
 %%%%%%%%%%%%compute deltaJB
   for pp=1:size(HS)
        deltaJB=deltaJB+2*(sum(diag(HS(pp,:)-HT(pp,:))*diag(HS(pp,:))*diag(1-HS(pp,:)))-...
                                                        sum(diag(HS(pp,:)-HT(pp,:))*diag(HT(pp,:))*diag(1-HT(pp,:))))/(size(HS,1)*HPHQ);

        deltaJB=deltaJB+2*(sum(diag(HS_r(pp,:)-HT_r(pp,:))*diag(HS_r(pp,:))*diag(1-HS_r(pp,:)))-...
                                                        sum(diag(HS_r(pp,:)-HT_r(pp,:))*diag(HT_r(pp,:))*diag(1-HT_r(pp,:))))/(size(HT_r,1)*HPHQ_r);
                                                 
   end

   for pp=1:size(HD)
        deltaJB=deltaJB-2*(sum(diag(HD(pp,:)-HE(pp,:))*diag(HD(pp,:))*diag(1-HD(pp,:)))-...
                                                        sum(diag(HD(pp,:)-HE(pp,:))*diag(HE(pp,:))*diag(1-HE(pp,:))))/(size(HD,1)*HDHE);
        deltaJB=deltaJB-2*(sum(diag(HD_r(pp,:)-HE_r(pp,:))*diag(HD_r(pp,:))*diag(1-HD_r(pp,:)))-...
                                                        sum(diag(HD_r(pp,:)-HE_r(pp,:))*diag(HE_r(pp,:))*diag(1-HE_r(pp,:))))/(size(HE_r,1)*HDHE_r);
   end
   
    vishidinc = momentum*vishidinc + ...
                epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
    visbiasinc = momentum*visbiasinc + (epsilonvb/numcases)*(posvisact-negvisact);
    hidbiasinc = momentum*hidbiasinc + (epsilonhb/numcases)*(poshidact-neghidact);
    vishid = vishid + mu*vishidinc+(1-mu)*detaW;
    visbiases = visbiases + mu*visbiasinc;
    hidbiases = hidbiases + mu*hidbiasinc+(1-mu)*deltaJB;
 end 
  fprintf(1, 'epoch %4i error %6.1f  \n', epoch, errsum); 
   error_return(1,epoch)=errsum;
end;

 poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));   
 hid = poshidprobs;
 vis = negdata;
 errorS=error_return;

 


  


