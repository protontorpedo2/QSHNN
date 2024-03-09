e=exp(1);% bottom of natural logarithm

w = [0.03,0.1,0.01,0.06];% initial weight quaternion

g = @(x)(e^(x)-e^(-x))/e^(-x)+e^x;% activation function applied
f = @(t,x)[w(1)*g(x(1))-w(2)*g(x(2))-w(3)*g(x(3))-w(4)*g(x(4))+1; 
    w(2)*g(x(1))+w(1)*g(x(2))-w(4)*g(x(3))+w(3)*g(x(4))+2;
    w(3)*g(x(1))+w(4)*g(x(2))+w(1)*g(x(3))-w(2)*g(x(4))+3; 
    w(4)*g(x(1))-w(3)*g(x(2))+w(2)*g(x(3))+w(1)*g(x(4))+2];
%nonlinear model of neural network(right hand side of the equation system)

q_0=[1,2,4,3];% initial value of neural network
u = 0.03;% learning rate constant
q = q_0;% output value of neural network
d = [0.06,0.12,0.34,0.5];% desired learning target
MSE = 1;% mean square error
period = 0;% training loop times
data_loss = [];
data_1 = [];data_2 = [];data_3 = [];data_4 = [];

while MSE > 0.001
    %mean square error between result and desire value
    MSE = sqrt(sum((d-q).^2));
    %disp(MSE);
    
    %visualize recorded data
    data_loss = [data_loss,MSE];
    data_1 = [data_1;q(1)];
    data_2 = [data_2;q(2)];
    data_3 = [data_3;q(3)];
    data_4 = [data_4;q(4)];
    
    % iterative updating
    w = (w'+u.*(matrix(d-q)*([q(1),-q(2),-q(3),-q(4)]')))';
    %operation result of updating network
    q = nt(q,w);
    
    %training period calculator
    period = period+1;
end

plot(data_loss,'linewidth',1.5);
grid on;
%plot(data_2(30:end));
figure;
plot3(data_1(2:end),data_2(2:end),data_3(2:end),'linewidth',1.5);
grid on;
figure;
plot3(data_1(2:end),data_2(2:end),data_4(2:end),'linewidth',1.5);
grid on;
figure;
plot3(data_2(2:end),data_3(2:end),data_4(2:end),'linewidth',1.5);
grid on;


% operation result of neural network
function p = nt(q_0,w)
    e=exp(1);% bottom of natural logarithm
    g = @(x)(e^(x)-e^(-x))/e^(-x)+e^x;% activation function applied
    f = @(t,x)[w(1)*g(x(1))-w(2)*g(x(2))-w(3)*g(x(3))-w(4)*g(x(4))+1; 
        w(2)*g(x(1))+w(1)*g(x(2))-w(4)*g(x(3))+w(3)*g(x(4))+2;
        w(3)*g(x(1))+w(4)*g(x(2))+w(1)*g(x(3))-w(2)*g(x(4))+3;
        w(4)*g(x(1))-w(3)*g(x(2))+w(2)*g(x(3))+w(1)*g(x(4))+2];
    opts = odeset('RelTol',1e-100,'AbsTol',1e-100);
    [t,y] = ode45(f,[0,5],q_0,opts);
    p = [y(end-2,1) y(end-2,2) y(end-2,3) y(end-2,4)];
end

% transfer the algorithm form of quaternion
function W = matrix(w)
   W = [w(1) -w(2) -w(3) -w(4); 
            w(2) w(1) -w(4) w(3);
            w(3) w(4) w(1) -w(2);
            w(4) -w(3) w(2) w(1)];% Matrix representation
end
