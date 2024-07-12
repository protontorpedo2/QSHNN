e = 3;%bottom of natural logarithm
n=2;%number of neurons

q0 = [0,0.2,0.4,0.3];% initial value of neurons
q_0 = [];
for i=1:n
    q_0 = [q_0,q0];
end 

w0 = [0,0.01,0.01,0.06];%initial weight values
w = ones(n^2,4);
for i=1:n^2
    w(i,:) = w0.*(i/(n^2));%randow value generator
% for i = 1:n
%     a = (i-1)*4;
%     for j = 1:n
%         b = (j-1)*4;
%         w(a+1:a+4,b+1) = multiplication(q(a+1:a+4),q(j+1:j+4)); 
%     end    
end

W=[];%adjacent matrix
L = [];
for i = 1:n
    for j = 1:n
        L =  [L,matrix(w(n*(i-1)+j,:))];
    end
    W = [W;L];
    L = [];
end

u = 0.05;% learning rate
q = q_0;

d0 = [0.5,0.5,0.5,0.5];
d = [];
for i=1:n
    if i == 2
        d = [d,2*d0];
    else   
        d = [d,d0];%training desire
    end
end 

MSE = 1;% mean square error
period = 0;% training loop times
data_loss = [];
data_1 = [];data_2 = [];data_3 = [];data_4 = [];
data_5 = [];data_6 = [];data_7 = [];data_8 = [];

syms  t g h
x = sym('x',[n*4,1]);
g = (e^(t)-e^(-t))/(e^(t)+e^t); % hyperbolic activation function

for i = 1:n*4
    x(i) = subs(g,x(i));
end

while MSE > 0.001
    
    MSE = sqrt(sum((d-q).^2));%mean square error between result and desire value
    disp('MSE:');
    disp(MSE);
     %visualize recorded data
    data_loss = [data_loss,MSE];

    delta = d-q;%[1,4*n] array, also q and d
    %disp(delta);

    
    data_1 = [data_1;q(1)]; %#ok<AGROW>
    data_2 = [data_2;q(2)]; %#ok<AGROW>
    data_3 = [data_3;q(3)]; %#ok<AGROW>
    data_4 = [data_4;q(4)];%#ok<AGROW>
    data_5 = [data_5;q(5)];%#ok<AGROW>
    data_6 = [data_6;q(6)];%#ok<AGROW>
    data_7 = [data_7;q(7)];%#ok<AGROW>
    data_8 = [data_8;q(8)];%#ok<AGROW>
    
    % iterative updating
    for i = 1:n
        a = (i-1)*4;
        for j = 1:n
            %update W_ij, neuron_j influence neuron_i through link W_ij.
            b = (j-1)*4;
            W(a+1:a+4,b+1) = (W(a+1:a+4,b+1)+u.*(matrix(delta(a+1:a+4))*([0,-q(b+2),-q(b+3),-q(b+4)].'))).';
            W(a+1:a+4,b+1:b+4) = matrix(W(a+1:a+4,b+1));
        end
    end
    %disp(W);

    h = W*x;
    vpa(h,5);
    f = char(h);
    f3 = strcat('@(t,x)',f);%combine two char rings
    
    for i = 4*n:-1:1
        num = num2str(i);
        str = strcat('x',num);
        str2 = strcat('x(',num,')');
        f3 = strrep(f3,str,str2);
        %f3 = strrep(f3,'x12','x(12)');
    end
    %disp(f3);
    f3 = str2func(f3);
    q = networkOperate(q_0,f3);%new result of updated network
    period = period+1;%training period calculator
    disp('period:');
    disp(period);
    % disp('Differential solver succeed!');
    % disp(q);
end

figure;
plot(data_loss(2:end),'linewidth',1.5);
grid on;
figure;
hold on;
x = 1:length(data_loss);
plot(x,data_1(1:end),'r',x,data_2(1:end),'r',x,data_3(1:end),'r',x,data_4(1:end),'r');
plot(x,data_5(1:end),'b',x,data_6(1:end),'b',x,data_7(1:end),'b',x,data_8(1:end),'b');
hold off;
grid on;
%clear;

% operation result of neural network
function q = networkOperate(q_0,f3)
    opts = odeset('RelTol',1e-10,'AbsTol',1e-100,'MaxStep',0.05);
    [~,y] = ode45(f3,[0,10],q_0,opts);
    q = y(end,:);
end

% Matrix representation
function Q = matrix(q)
   Q = [q(1) -q(2) -q(3) -q(4); 
           q(2) q(1) -q(4) q(3);
           q(3) q(4) q(1) -q(2);
           q(4) -q(3) q(2) q(1)];
end


function Q = multiplication(p,q)
    Q = (matrix(p)*q.').';
end
