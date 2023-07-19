import {Callback} from 'element-plus';

interface Point {
  x: number;
  y: number;
}

export class Draw {
  private c: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private mouseDown: boolean;
  private readonly color: string;
  private readonly size: number;
  private prePoint: Point | null = null;
  private readonly changeCallback: Callback | null;

  private isLeftDown = false

  /**
   * is clear state
   * clear means did not draw anything
   */
  isClear = true

  constructor(canvas: HTMLCanvasElement,
              changeCallback: Callback | null = null,
              color='#FFFFFF', size=15) {
    this.changeCallback = Function;
    this.c = canvas
    this.color = color
    this.size = size
    this.ctx = this.c!.getContext('2d') as CanvasRenderingContext2D;

    this.mouseDown = false;

    this.setSize();

    this.listen();

    this.clear();
  }

  listen(){
    this.c.addEventListener('mousedown', (e)=>{
      if (e.button !== 0) {
        return
      }
      this.prePoint = {x: e.offsetX, y: e.offsetY}
      this.changeCallback?.('confirm', 'confirm')
      this.isLeftDown = true
    });

    this.c.addEventListener('mouseup', ()=>{
      this.prePoint = null;
      if (this.isLeftDown) {
        this.setClear(false)
        this.isLeftDown = false
      }
    });

    this.c.addEventListener('mouseleave', ()=>{
      this.prePoint = null;
    });

    this.c.addEventListener('mousemove', (e)=>{
      if (e.button !== 0) {
        return
      }
      this.draw({x: e.offsetX, y: e.offsetY})
    });

    window.addEventListener('resize', ()=>{
      this.setSize();
      this.clear();
    });
  }

  setSize(){
    this.c.width = this.c.offsetWidth
    this.c.height = this.c.offsetHeight
  }

  clear(){
    this.ctx.clearRect(0, 0, this.c.width, this.c.height);
    this.setClear(true)
  }

  setClear(isClear: boolean) {
    console.log('setClear: ', isClear)
    this.isClear = isClear
  }

  draw(point: Point){
    if (!this.prePoint) {
      return
    }

    this.ctx.lineCap = 'round';
    this.ctx.lineJoin="round";

    this.ctx.strokeStyle = this.color;
    this.ctx.lineWidth = this.size;

    this.ctx.beginPath();
    this.ctx.moveTo(this.prePoint.x, this.prePoint.y);
    this.ctx.lineTo(point.x, point.y);
    this.ctx.stroke();
    this.ctx.closePath();
    this.prePoint = point
  }
}
