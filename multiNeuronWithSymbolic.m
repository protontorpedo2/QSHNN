e = 3;%exp(1);% bottom of natural logarithm
n=8;%number of neurons

w0 = [0.03,0.01,0.01,0.06];%initial weight values
w = ones(n^2,4);
for i=1:n^2
    w(i,:) = w0.*(i/(n^2));%randow initial value generator
end

W=[];
L = [];
for i = 1:n
    for j = 1:n
        L =  [L,matrix(w(n*(i-1)+j,:))];
    end
    W = [W;L];
    L = [];
end
%display(W);%W is linear forward transmission matrix

q0 = [0.1,0.2,0.4,0.3];% initial value of neurons
q_0 = [];
for i=1:n
    q_0 = [q_0,q0.*(5*n^(1/3))];%randow initial value generator
end 
%display(q_0);

u = 0.05;% learning rate constant
q = q_0;
%disp(q);

d0 = [0.06,0.12,0.34,0.5];
d = [];
for i=1:n
    d = [d,d0];%training desire
end 
%display(d);

MSE = 1;% mean square error
period = 0;% training loop times
data_loss = [];

syms x t g h
x = sym('x',[n*4,1]);
%g = (e^(t)-e^(-t))/e^(t)+e^t; % hyperbolic activation function
g = t;%linear activation function
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
    
    % iterative updating
    for i = 1:n
        a = (i-1)*4;
        for j = 1:n
            %update W_ij, neuron_j influence neuron_i through link W_ij.
            b = (j-1)*4;
            W(a+1:a+4,b+1) = (W(a+1:a+4,b+1)+u.*(matrix(delta(a+1:a+4))*([q(b+1),-q(b+2),-q(b+3),-q(b+4)].'))).';
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

plot(data_loss,'linewidth',1.5);
grid on;

% operation result of neural network
function q = networkOperate(q_0,f3)
    opts = odeset('RelTol',1e-13,'AbsTol',1e-100,'MaxStep',0.001);
    [~,y] = ode45(f3,[0,5],q_0,opts);
    q = y(end-1,:);
end

% Matrix representation
function Q = matrix(q)
   Q = [q(1) -q(2) -q(3) -q(4); 
           q(2) q(1) -q(4) q(3);
           q(3) q(4) q(1) -q(2);
           q(4) -q(3) q(2) q(1)];
end
